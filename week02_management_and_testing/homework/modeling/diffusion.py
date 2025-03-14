from typing import Dict, Tuple

import torch
import torch.nn as nn


class DiffusionModel(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        num_timesteps: int,
    ):
        super().__init__()
        self.eps_model = eps_model

        for name, schedule in get_schedules(betas[0], betas[1], num_timesteps).items():
            self.register_buffer(name, schedule)

        self.num_timesteps = num_timesteps
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        timestep = torch.randint(1, self.num_timesteps + 1, (x.shape[0],), device=x.device)
        eps = torch.randn_like(x)
        
        x_t = (
            self.sqrt_alphas_cumprod[timestep, None, None, None] * x
            + self.sqrt_one_minus_alpha_prod[timestep, None, None, None] * eps
        )
        return self.criterion(eps, self.eps_model(x_t, timestep / self.num_timesteps))

    def sample(self, num_samples: int, size, device) -> torch.Tensor:

        x_i = torch.randn(num_samples, *size, device=device)

        for i in range(self.num_timesteps, 0, -1):
            z = torch.randn(num_samples, *size, device=device) if i > 1 else 0
            eps = self.eps_model(x_i, torch.tensor(i / self.num_timesteps).repeat(num_samples, 1).to(device))

            mu = self.sqrt_alphas[i] / self.one_minus_alpha_prod[i] * (
                self.one_minus_alpha_prod[i - 1] * x_i + (x_i - eps * self.sqrt_one_minus_alpha_prod[i]) * self.betas[i]
            )
            std = self.sqrt_betas[i]
            x_i = mu + std * z

        return x_i


def get_schedules(beta1: float, beta2: float, num_timesteps: int) -> Dict[str, torch.Tensor]:
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    betas = (beta2 - beta1) * torch.arange(0, num_timesteps + 1, dtype=torch.float32) / num_timesteps + beta1
    sqrt_betas = torch.sqrt(betas)
    alphas = 1 - betas
    sqrt_alphas = torch.sqrt(alphas)

    alphas_cumprod = torch.cumprod(alphas, dim=0)
    one_minus_alpha_prod = 1 - alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "sqrt_betas": sqrt_betas,
        "sqrt_alphas": sqrt_alphas,
        "alphas_cumprod": alphas_cumprod,
        "one_minus_alpha_prod": one_minus_alpha_prod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alpha_prod": sqrt_one_minus_alpha_prod,
    }
