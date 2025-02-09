import os
import tempfile
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, train_epoch, generate_samples
from modeling.unet import UnetModel


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # suspicious normalization values
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


# Test train_epoch with limited batches
class LimitedDataLoader:
    def __init__(self, dataloader, num_batches):
        self.dataloader = dataloader
        self.num_batches = num_batches
    
    def __iter__(self):
        iterator = iter(self.dataloader)
        for _ in range(self.num_batches):
            try:
                yield next(iterator)
            except StopIteration:
                return


@pytest.mark.parametrize(
    ["device", "num_timesteps", "hidden_size", "batch_size", "num_batches", "expected_loss"], 
    [
        ["cpu", 100, 32, 4, 50, 0.5],  # Smaller model, fewer steps for CPU
        pytest.param(
            "cuda", 1000, 64, 64, None, 0.5,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
        ),
    ]
)
def test_training(device, train_dataset, num_timesteps, hidden_size, batch_size, num_batches, expected_loss):
    # Initialize model with given parameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=hidden_size),
        betas=(1e-4, 0.02),
        num_timesteps=num_timesteps,
    )
    ddpm.to(device)

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if device == "cpu":
        dataloader = LimitedDataLoader(dataloader, num_batches)

    optimizer = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    
    # Test train_step
    x, _ = next(iter(dataloader))
    initial_loss = train_step(ddpm, x, optimizer, device)
    assert not torch.isnan(initial_loss), "Initial training step produced NaN loss"

    train_epoch(ddpm, dataloader, optimizer, device, debug=True)

    # Test final training step to verify loss improvement
    final_loss = train_step(ddpm, x, optimizer, device)
    assert final_loss < initial_loss, "Training did not reduce loss"
    assert final_loss < expected_loss, f"Final loss {final_loss} exceeds expected maximum {expected_loss}"

    # Test sample generation
    samples = ddpm.sample(2, (3, 32, 32), device)
    assert samples.shape == (2, 3, 32, 32), f"Expected shape (2, 3, 32, 32), got {samples.shape}"
    assert not torch.isnan(samples).any(), "Generated samples contain NaN values"
    
    # Test generate_samples function (with temporary file)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        generate_samples(ddpm, device, tmp.name, 2, (3, 32, 32), debug=True)
        assert os.path.exists(tmp.name), "Sample image was not generated"
        assert os.path.getsize(tmp.name) > 0, "Generated image is empty"
        # TODO: check if the image is normalized
        os.unlink(tmp.name)
