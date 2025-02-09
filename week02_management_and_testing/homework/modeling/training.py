import hydra
import wandb
from omegaconf import DictConfig
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from modeling.diffusion import DiffusionModel


def train_step(model: DiffusionModel, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, device: str, debug: bool = False):
    model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    for batch_idx, (x, _) in enumerate(pbar):
        train_loss = train_step(model, x, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")

        # Log metrics to wandb
        if not debug:
            wandb.log({
                "train_loss": train_loss.item(),
                "loss_ema": loss_ema.item(),
                "learning_rate": optimizer.param_groups[0]["lr"]
            })

            if batch_idx == 0:
                wandb.log({
                    "train_batch": wandb.Image(
                        make_grid(x, nrow=4, normalize=True, value_range=(-1, 1))
                    )
                })


def generate_samples(model: DiffusionModel, device: str, path: str, num_samples: int, image_size: tuple[int, int, int], debug: bool = False):
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, image_size, device=device)
        grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
        save_image(grid, path)
        if not debug:
            wandb.log({
                "samples": wandb.Image(
                    grid
                )
            })
