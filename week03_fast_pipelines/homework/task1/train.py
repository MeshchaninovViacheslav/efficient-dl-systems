import torch
from torch import nn
from tqdm.auto import tqdm
from argparse import ArgumentParser
from unet import Unet

from dataset import get_train_data


class GradScaler:
    def __init__(self, loss_scaling, init_scale=2**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=100):
        self.loss_scaling = loss_scaling
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.steps_since_inc = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_grads(self, model):
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.div_(self.scale)

    def check_overflow(self, model):
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return True
        return False

    def update(self, model):
        if self.loss_scaling == "static":
            return True
        if self.check_overflow(model):
            self.scale *= self.backoff_factor
            self.steps_since_inc = 0
            return False  
        else:
            self.steps_since_inc += 1
            if self.steps_since_inc >= self.growth_interval:
                self.scale *= self.growth_factor
                self.steps_since_inc = 0
            return True  


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
) -> None:
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

            scaled_loss = scaler.scale_loss(loss)
            scaled_loss.backward()
            scaler.unscale_grads(model)
            if not scaler.update(model):
                continue  
            optimizer.step()

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train(loss_scaling: str):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler(loss_scaling)

    train_loader = get_train_data()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, device=device, scaler=scaler)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--loss_scaling", type=str, default="static")
    args = parser.parse_args()
    train(args.loss_scaling)
