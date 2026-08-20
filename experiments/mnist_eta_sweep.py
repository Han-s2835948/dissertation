from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.utils import make_grid, save_image

from mnist_cdg.checkpoint import load_weights
from mnist_cdg.common import load_config, resolve_device, seed_everything
from mnist_cdg.models import HNet, MNISTClassifier, ScoreUNet
from mnist_cdg.sde import VPSDE, reverse_sde_sample


def tensor_to_pil(grid: torch.Tensor) -> Image.Image:
    grid = grid.detach().cpu().clamp(0, 1)
    if grid.shape[0] == 1:
        array = (grid[0].numpy() * 255).round().astype("uint8")
        return Image.fromarray(array, mode="L").convert("RGB")
    array = (grid.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def save_rows(rows: list[tuple[str, torch.Tensor]], output: Path, count: int) -> None:
    rendered: list[tuple[str, Image.Image]] = []
    for label, images in rows:
        display = (images[:count].cpu() + 1) / 2
        grid = make_grid(display, nrow=count, padding=2, pad_value=1.0)
        rendered.append((label, tensor_to_pil(grid)))
    label_width = 76
    row_gap = 5
    width = label_width + max(image.width for _, image in rendered)
    height = sum(image.height for _, image in rendered) + row_gap * (len(rendered) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for label, image in rendered:
        draw.text((6, y + max(0, (image.height - 11) // 2)), label, fill="black")
        canvas.paste(image, (label_width, y))
        y += image.height + row_gap
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def class_effective_count(predictions: torch.Tensor) -> float:
    histogram = torch.bincount(predictions.cpu(), minlength=10).float()
    probabilities = histogram / histogram.sum()
    nonzero = probabilities[probabilities > 0]
    entropy = -(nonzero * nonzero.log()).sum().item()
    return math.exp(entropy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_vp_formal.yaml")
    parser.add_argument("--score", default="outputs_formal/mnist/score/latest.pt")
    parser.add_argument("--h-checkpoint", default="outputs_formal/mnist/h/classifier_7.pt")
    parser.add_argument("--classifier", default="outputs_formal/mnist/classifier/latest.pt")
    parser.add_argument("--etas", nargs="+", type=float,
                        default=[0, 1, 2, 4, 8, 16, 24, 32, 64])
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=7042)
    parser.add_argument("--display-count", type=int, default=16)
    parser.add_argument("--output-dir", default="outputs_formal/mnist/eta_sweep_classifier_7_seed7042")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    steps = args.steps or cfg["sampling"]["steps"]
    target = cfg["h_training"]["target_class"]
    h_floor = cfg["h_training"]["h_floor"]
    guidance_clip = cfg["h_training"]["guidance_clip"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    score = ScoreUNet(**cfg["model"]).to(device).eval()
    hnet = HNet(**cfg["model"]).to(device).eval()
    classifier = MNISTClassifier().to(device).eval()
    load_weights(args.score, score, device)
    load_weights(args.h_checkpoint, hnet, device)
    load_weights(args.classifier, classifier, device)
    sde = VPSDE(**cfg["sde"])

    def make_guidance(eta: float):
        def guidance(tau: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                xin = x.detach().requires_grad_(True)
                h = hnet(xin, tau).clamp_min(h_floor)
                grad = eta * torch.autograd.grad(torch.log(h).sum(), xin)[0]
                flat = grad.flatten(1)
                norm = flat.norm(dim=1).clamp_min(1e-8)
                clip_scale = (guidance_clip / norm).clamp(max=1.0)
                return (grad * clip_scale[:, None, None, None]).detach()
        return guidance

    images_by_eta: dict[float, torch.Tensor] = {}
    metrics: dict[str, object] = {
        "samples": args.samples,
        "steps": steps,
        "seed": args.seed,
        "target_class": target,
        "paired_randomness": True,
        "score_checkpoint": args.score,
        "h_checkpoint": args.h_checkpoint,
        "classifier_checkpoint": args.classifier,
        "results": {},
    }
    baseline = None
    baseline_success = None
    for eta in args.etas:
        seed_everything(args.seed)
        guidance = None if eta == 0 else make_guidance(eta)
        samples = reverse_sde_sample(
            score, sde, (args.samples, 1, 28, 28), steps, device,
            guidance_fn=guidance).cpu()
        with torch.no_grad():
            probabilities = classifier(samples.to(device)).softmax(1).cpu()
        predictions = probabilities.argmax(1)
        accepted = predictions.eq(target)
        if eta == 0:
            baseline = samples
            baseline_success = accepted
        assert baseline is not None and baseline_success is not None
        gained = ((~baseline_success) & accepted).sum().item()
        lost = (baseline_success & (~accepted)).sum().item()
        result = {
            "success_count": int(accepted.sum().item()),
            "success_rate": accepted.float().mean().item(),
            "mean_target_probability": probabilities[:, target].mean().item(),
            "predicted_class_histogram": torch.bincount(predictions, minlength=10).tolist(),
            "predicted_class_effective_count": class_effective_count(predictions),
            "pixel_saturation_rate": (((samples <= -0.98) | (samples >= 0.98)).float().mean().item()),
            "mean_absolute_change_from_baseline": (samples - baseline).abs().mean().item(),
            "gained_vs_unconditional": int(gained),
            "lost_vs_unconditional": int(lost),
        }
        metrics["results"][f"eta_{eta:g}"] = result
        images_by_eta[eta] = samples
        torch.save(samples, output / f"samples_eta_{eta:g}.pt")
        save_image((samples + 1) / 2, output / f"grid_eta_{eta:g}.png",
                   nrow=max(1, int(math.sqrt(args.samples))))
        print(f"eta={eta:g} success={result['success_rate']:.2%} "
              f"mean_p7={result['mean_target_probability']:.4f}")

    save_rows([("unconditional", images_by_eta[0.0])],
              output / "unconditional_same_noise.png", args.display_count)
    conditional_rows = [(f"eta={eta:g}", images_by_eta[eta])
                        for eta in args.etas if eta != 0]
    save_rows(conditional_rows, output / "conditional_eta_comparison_same_noise.png",
              args.display_count)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"saved MNIST eta sweep to {output}")


if __name__ == "__main__":
    main()
