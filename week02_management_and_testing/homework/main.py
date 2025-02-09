import os
import hydra
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from omegaconf import OmegaConf, DictConfig
import wandb
import random
import numpy as np

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel


def seed_everything(seed: int):
    """
    Set seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The random seed value.
    """
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy's random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (single)
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (all devices)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables auto-optimizations

    print(f"🔢 Random seed set to: {seed}")


def get_dataloader(cfg):
    """
    Creates and returns a DataLoader for the CIFAR-10 dataset with augmentation.

    Args:
        cfg (DictConfig): Hydra config containing dataset parameters.

    Returns:
        DataLoader: PyTorch DataLoader for training and validation sets.
    """
    transform_list = [
        transforms.RandomHorizontalFlip() if cfg.dataset.augmentation else transforms.Lambda(lambda x: x),  # Apply flip if enabled
        transforms.ToTensor(),
        transforms.Normalize(cfg.dataset.mean, cfg.dataset.std)  # Normalize to [-1, 1]
    ]

    transform = transforms.Compose(transform_list)

    # Load dataset
    train_dataset = CIFAR10(
        root=cfg.dataset.name,
        train=True,
        download=True,
        transform=transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    print(f"📊 Loaded CIFAR-10 dataset with augmentation: {cfg.dataset.augmentation}")
    return train_loader


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    exp_name = f"{cfg.model.name}-{cfg.optimizer.name}-{cfg.training.batch_size}-{cfg.training.num_epochs}-{cfg.optimizer.lr}"
    wandb.init(project=cfg.project.name, config=cfg, name=exp_name)

    # Save Hydra config to a YAML file
    config_path = "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Log the config file as an artifact in W&B
    artifact = wandb.Artifact(name="hydra_config", type="config")
    artifact.add_file(config_path)
    wandb.log_artifact(artifact)

    print("📂 Hydra config logged to W&B as an artifact!")

    seed_everything(cfg.project.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ddpm = DiffusionModel(
        eps_model=UnetModel(
            in_channels=cfg.model.input_channels, 
            out_channels=cfg.model.output_channels, 
            hidden_size=cfg.model.hidden_size
        ),
        betas=cfg.diffusion.scheduler,
        num_timesteps=cfg.diffusion.num_timesteps,
    )
    ddpm.to(device)

    dataloader = get_dataloader(cfg)
    
    # Define optimizer
    if cfg.optimizer.name == "adam":
        optimizer = torch.optim.Adam(ddpm.parameters(), lr=cfg.optimizer.lr,
                                     betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
                                     weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(ddpm.parameters(), lr=cfg.optimizer.lr,
                                    momentum=cfg.optimizer.momentum,
                                    weight_decay=cfg.optimizer.weight_decay)

    os.makedirs(cfg.training.samples_dir, exist_ok=True)
    for i in range(cfg.training.num_epochs):
        train_epoch(ddpm, dataloader, optimizer, device)
        generate_samples(ddpm, device, path=f"{cfg.training.samples_dir}/{i:03d}.png", 
                         num_samples=cfg.training.num_samples, image_size=cfg.dataset.image_size)


if __name__ == "__main__":
    main()
