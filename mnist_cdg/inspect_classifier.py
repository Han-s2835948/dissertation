from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

from .checkpoint import load_weights
from .common import ensure_dir, load_config, resolve_device, seed_everything
from .models import HNet, MNISTClassifier, ScoreUNet
from .sde import VPSDE, reverse_sde_sample


def classify(model: MNISTClassifier, images: torch.Tensor):
    with torch.no_grad():
        probabilities = model(images).softmax(dim=1).cpu()
    confidence, prediction = probabilities.max(dim=1)
    second_confidence, second_prediction = probabilities.topk(2, dim=1).values[:, 1], probabilities.topk(2, dim=1).indices[:, 1]
    return prediction, confidence, second_prediction, second_confidence, probabilities


def save_audit(
    images: torch.Tensor,
    predictions: torch.Tensor,
    confidence: torch.Tensor,
    second_predictions: torch.Tensor,
    second_confidence: torch.Tensor,
    probabilities: torch.Tensor,
    output_dir: Path,
    name: str,
    true_labels: torch.Tensor | None = None,
    columns: int = 8,
):
    image_dir = ensure_dir(output_dir / f"{name}_individual")
    font = ImageFont.load_default()
    cell_w, cell_h = 116, 128
    rows = (len(images) + columns - 1) // columns
    grid = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    csv_path = output_dir / f"{name}.csv"

    fieldnames = ["sample", "image_file", "true_label", "predicted_label", "confidence",
                  "second_label", "second_confidence"] + [f"prob_{i}" for i in range(10)]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, image in enumerate(images.cpu()):
            display = to_pil_image(((image + 1.0) / 2.0).clamp(0, 1)).resize((84, 84))
            image_name = f"sample_{i:03d}.png"
            display.save(image_dir / image_name)
            pred = int(predictions[i])
            conf = float(confidence[i])
            true = "" if true_labels is None else int(true_labels[i])
            if true_labels is None:
                title = f"#{i:03d}  pred={pred}"
                colour = "black"
            else:
                title = f"#{i:03d} true={true} pred={pred}"
                colour = "darkgreen" if pred == true else "red"
            panel = Image.new("RGB", (cell_w, cell_h), "white")
            panel.paste(display, ((cell_w - 84) // 2, 3))
            draw = ImageDraw.Draw(panel)
            draw.text((4, 91), title, fill=colour, font=font)
            draw.text((4, 106), f"confidence={conf:.1%}", fill=colour, font=font)
            grid.paste(panel, ((i % columns) * cell_w, (i // columns) * cell_h))

            row = {
                "sample": i,
                "image_file": str(image_dir / image_name),
                "true_label": true,
                "predicted_label": pred,
                "confidence": f"{conf:.6f}",
                "second_label": int(second_predictions[i]),
                "second_confidence": f"{float(second_confidence[i]):.6f}",
            }
            row.update({f"prob_{digit}": f"{float(probabilities[i, digit]):.6f}" for digit in range(10)})
            writer.writerow(row)

    grid_path = output_dir / f"{name}_labelled.png"
    grid.save(grid_path)
    print(f"saved labelled grid: {grid_path}")
    print(f"saved prediction table: {csv_path}")
    print(f"saved individual images: {image_dir}")


def balanced_real_examples(data_dir: str, per_class: int):
    dataset = datasets.MNIST(data_dir, train=False, download=False, transform=transforms.ToTensor())
    buckets: list[list[torch.Tensor]] = [[] for _ in range(10)]
    for image, label in dataset:
        if len(buckets[label]) < per_class:
            buckets[label].append(image * 2.0 - 1.0)
        if all(len(bucket) == per_class for bucket in buckets):
            break
    images, labels = [], []
    for label, bucket in enumerate(buckets):
        images.extend(bucket)
        labels.extend([label] * len(bucket))
    return torch.stack(images), torch.tensor(labels)


def main():
    parser = argparse.ArgumentParser(description="Create human-checkable MNIST classifier reports.")
    parser.add_argument("--config", default="configs/mnist_vp.yaml")
    parser.add_argument("--classifier", default="outputs_quick/classifier/latest.pt")
    parser.add_argument("--score", default="outputs_quick/score/latest.pt")
    parser.add_argument("--h-checkpoint", default="outputs_quick/h/class_7.pt")
    parser.add_argument("--output", default="outputs_quick/classifier_audit")
    parser.add_argument("--mode", choices=["real", "unconditional", "conditional", "all"], default="all")
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--generated-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    output_dir = ensure_dir(args.output)
    classifier = MNISTClassifier().to(device).eval()
    load_weights(args.classifier, classifier, device)

    if args.mode in {"real", "all"}:
        images, labels = balanced_real_examples(cfg["data_dir"], args.per_class)
        result = classify(classifier, images.to(device))
        save_audit(images, *result, output_dir, "real_mnist", true_labels=labels)
        accuracy = (result[0] == labels).float().mean().item()
        print(f"real MNIST audit accuracy: {accuracy:.2%} ({len(labels)} images)")

    if args.mode in {"unconditional", "conditional", "all"}:
        score = ScoreUNet(**cfg["model"]).to(device).eval()
        load_weights(args.score, score, device)
        sde = VPSDE(**cfg["sde"])

        if args.mode in {"unconditional", "all"}:
            samples = reverse_sde_sample(score, sde, (args.generated_samples, 1, 28, 28),
                                         args.steps, device)
            result = classify(classifier, samples)
            save_audit(samples, *result, output_dir, "generated_unconditional")

        if args.mode in {"conditional", "all"}:
            hc = cfg["h_training"]
            hnet = HNet(**cfg["model"]).to(device).eval()
            load_weights(args.h_checkpoint, hnet, device)

            def guidance(tau, x):
                with torch.enable_grad():
                    xin = x.detach().requires_grad_(True)
                    h = hnet(xin, tau).clamp_min(hc["h_floor"])
                    grad = torch.autograd.grad(torch.log(h).sum(), xin)[0]
                    flat = grad.flatten(1)
                    norm = flat.norm(dim=1).clamp_min(1e-8)
                    scale = (hc["guidance_clip"] / norm).clamp(max=1.0)
                    return (grad * scale[:, None, None, None]).detach()

            samples = reverse_sde_sample(score, sde, (args.generated_samples, 1, 28, 28),
                                         args.steps, device, guidance_fn=guidance)
            result = classify(classifier, samples)
            save_audit(samples, *result, output_dir, f"generated_conditional_class_{hc['target_class']}")
            success = (result[0] == hc["target_class"]).float().mean().item()
            print(f"conditional target-{hc['target_class']} rate: {success:.2%}")


if __name__ == "__main__":
    main()
