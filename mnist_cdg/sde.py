from __future__ import annotations

import torch


class VPSDE:
    """Variance-preserving SDE on forward data time t in [0, 1].

    dX_t = -0.5 beta(t) X_t dt + sqrt(beta(t)) dW_t.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, eps: float = 1e-3):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = eps

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    @staticmethod
    def _expand(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return v.reshape(v.shape[0], *([1] * (x.ndim - 1)))

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        log_alpha = -0.25 * (self.beta_max - self.beta_min) * t.square()
        log_alpha = log_alpha - 0.5 * self.beta_min * t
        return torch.exp(log_alpha)

    def marginal(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        noise = torch.randn_like(x0) if noise is None else noise
        alpha = self._expand(self.alpha(t), x0)
        std = torch.sqrt(torch.clamp(1.0 - alpha.square(), min=1e-12))
        return alpha * x0 + std * noise, noise, std

    def forward_drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * self._expand(self.beta(t), x) * x

    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._expand(torch.sqrt(self.beta(t)), x)

    def reverse_drift(self, t: torch.Tensor, x: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """Drift in generative time tau=1-t: -f(t,x)+g(t)^2 score(t,x)."""
        beta = self._expand(self.beta(t), x)
        return 0.5 * beta * x + beta * score


@torch.no_grad()
def reverse_sde_sample(
    score_model,
    sde: VPSDE,
    shape: tuple[int, ...],
    steps: int,
    device: torch.device,
    guidance_fn=None,
    return_trajectory: bool = False,
    save_steps: int = 51,
    return_random_pair: bool = False,
):
    """Euler-Maruyama sampler from noise (tau=0) to data (tau=1-eps)."""
    x = torch.randn(shape, device=device)
    dt = (1.0 - sde.eps) / steps
    saved_x, saved_tau = [], []
    save_indices = set(torch.linspace(0, steps, save_steps).round().long().tolist())
    if return_trajectory and return_random_pair:
        raise ValueError("return_trajectory and return_random_pair are mutually exclusive")
    if return_random_pair:
        pair_indices = torch.randint(0, max(steps - 1, 1), (shape[0],), device=device)
        pair_x = torch.empty_like(x)
        pair_x_next = torch.empty_like(x)

    for i in range(steps):
        tau_value = i * dt
        forward_t_value = 1.0 - tau_value
        t = torch.full((shape[0],), forward_t_value, device=device)
        tau = torch.full((shape[0],), tau_value, device=device)
        if return_trajectory and i in save_indices:
            saved_x.append(x.detach().cpu())
            saved_tau.append(tau_value)
        pair_mask = pair_indices.eq(i) if return_random_pair else None
        if return_random_pair and pair_mask.any():
            pair_x[pair_mask] = x[pair_mask].detach()
        # The score supplies the unconditional reverse drift.  Conditional
        # guidance is added separately so the same score model can be reused.
        score = score_model(x, t)
        drift = sde.reverse_drift(t, x, score)
        if guidance_fn is not None:
            drift = drift + sde._expand(sde.beta(t), x) * guidance_fn(tau, x)
        x = x + drift * dt
        if i < steps - 1:
            x = x + sde.diffusion(t, x) * (dt**0.5) * torch.randn_like(x)
        if return_random_pair and pair_mask.any():
            pair_x_next[pair_mask] = x[pair_mask].detach()

    if return_trajectory and steps in save_indices:
        saved_x.append(x.detach().cpu())
        saved_tau.append(1.0 - sde.eps)
    if return_trajectory:
        return x.clamp(-1, 1), torch.stack(saved_x, dim=1), torch.tensor(saved_tau)
    if return_random_pair:
        pair_tau = pair_indices.float() * dt
        return (x.clamp(-1, 1), pair_x.cpu(), pair_x_next.cpu(),
                pair_tau.cpu(), (pair_tau + dt).cpu())
    return x.clamp(-1, 1)
