from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, optimizer=None, **metadata):
    payload = {"model": model.state_dict(), **metadata}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_weights(path: str | Path, model, device, prefer_ema: bool = True):
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("ema") if prefer_ema else None
    model.load_state_dict(state if state is not None else payload["model"])
    return payload
