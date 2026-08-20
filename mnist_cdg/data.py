from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ToModelRange:
    def __init__(self, dequantize: bool = True):
        self.dequantize = dequantize

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.dequantize:
            x = (x * 255.0 + torch.rand_like(x)) / 256.0
        return x * 2.0 - 1.0


def mnist_loaders(data_dir: str, batch_size: int, num_workers: int = 2, dequantize: bool = True):
    transform = transforms.Compose([transforms.ToTensor(), ToModelRange(dequantize)])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train, shuffle=True, drop_last=True, **kwargs),
        DataLoader(test, shuffle=False, drop_last=False, **kwargs),
    )

