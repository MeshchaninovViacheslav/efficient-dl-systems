import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from functools import partial
import time
import random
import numpy as np

from syncbn import SyncBatchNorm

torch.set_num_threads(1)


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


def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


class Net(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self, args=None):
        super().__init__()

        self.implementation = args.implementation

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        if args.implementation == "custom":
            self.bn1 = SyncBatchNorm(128)
        elif args.implementation == "torch":
            self.bn1 = nn.SyncBatchNorm(128, affine=False)
        else:
            raise ValueError(f"Invalid implementation: {args.implementation}")

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def scatter_dataset(dataset, size, rank, device):
    total_samples = len(dataset)
    chunk_size = total_samples // size
    if rank == 0:
        new_total = chunk_size * size
        full_indices = torch.arange(new_total, dtype=torch.long, device=device)
        scatter_list = list(full_indices.view(size, chunk_size))
    else:
        scatter_list = None
    recv_indices = torch.empty(chunk_size, dtype=torch.long, device=device)
    dist.scatter(recv_indices, scatter_list=scatter_list, src=0)
    # Create a Subset of the test dataset using the received indices.
    subset = Subset(dataset, recv_indices.tolist())
    return subset


def get_loaders(rank, size, args=None, device=None):
    train_dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),   
            ]
        ),
        download=True,
        train=True,
    )
    test_dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=True,
        train=False,
    )
    train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset, size, rank), 
                              batch_size=args.batch_size, num_workers=10)

    test_dataset = scatter_dataset(test_dataset, size, rank, device)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=10)
    return train_loader, test_loader


def train_epoch(model_ddp, train_loader, optimizer, args, device, rank):
    model_ddp.train()
    epoch_loss = torch.zeros((1,), device=device)
    optimizer.zero_grad()
    
    if rank == 0:
        train_range = tqdm(train_loader)
    else:
        train_range = train_loader

    for step, (data, target) in enumerate(train_range):
        data = data.to(device)
        target = target.to(device)

        output = model_ddp(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        
        epoch_loss += loss.detach()
        if (step + 1) % args.grad_accumulation == 0:
            if args.implementation == "custom":
                average_gradients(model_ddp)
            
            optimizer.step()
            optimizer.zero_grad()

        acc = (output.argmax(dim=1) == target).float().mean()

        if rank == 0:
            train_range.set_postfix(loss=loss.item(), acc=acc.item())
        

def test_epoch(model_ddp, test_loader, args, device, rank):
    model_ddp.eval()
    # Test
    total_size = torch.tensor(0, device=device)
    total_loss = torch.tensor(0.0, device=device)
    total_acc = torch.tensor(0.0, device=device)
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        output = model_ddp(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        total_loss += loss.detach() * data.size(0)
        total_size += data.size(0)
        acc = (output.argmax(dim=1) == target).float().mean()
        total_acc += acc.detach() * data.size(0)

    # Aggregate metrics from all workers (only rank 0 gets the sum).
    dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_acc, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_size, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        avg_loss = total_loss.item() / total_size.item()
        accuracy = total_acc.item() / total_size.item()
        return avg_loss, accuracy
    else:
        return None, None


def measure_peak_memory(device):
    """
    Measure the peak GPU memory usage on the given device.
    Uses torch.cuda.max_memory_allocated and resets stats afterward.
    """
    torch.cuda.synchronize(device)
    peak_mem = torch.cuda.max_memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    return peak_mem


def get_avg_between_percentiles(values, lower_percentile, upper_percentile):
    sorted_values = sorted(values)
    lower_idx = int(len(sorted_values) * lower_percentile)
    upper_idx = int(len(sorted_values) * upper_percentile)
    sorted_values = sorted_values[lower_idx:upper_idx]
    if len(sorted_values) == 0:
        return 0
    return sum(sorted_values) / (upper_idx - lower_idx)


def run_experiment(model_ddp, train_loader, test_loader, optimizer, args, device, rank, num_epochs=10):
    all_epoch_times = []
    all_epoch_losses = []
    all_epoch_accs = []
    all_epoch_peaks = []
    for epoch in range(num_epochs):
        start_epoch = time.perf_counter()
        train_epoch(model_ddp, train_loader, optimizer, args, device, rank)
        end_epoch = time.perf_counter()
        epoch_duration = end_epoch - start_epoch
        peak_memory = measure_peak_memory(device)

        epoch_loss, epoch_acc = test_epoch(model_ddp, test_loader, args, device, rank)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Valid Loss = {epoch_loss:.4f}, "
                  f"Valid Acc = {epoch_acc:.4f}, "
                  f"Epoch Time = {epoch_duration:.2f}s, "
                  f"Peak Memory = {peak_memory/1e6:.2f} MB")
            all_epoch_times.append(epoch_duration)
            all_epoch_losses.append(epoch_loss)
            all_epoch_peaks.append(peak_memory)
            all_epoch_accs.append(epoch_acc)

    if rank == 0:
        # Final quality metric can be derived from the final epoch loss (or accuracy if measured).
        final_quality = all_epoch_accs[-1]
        avg_time = get_avg_between_percentiles(all_epoch_times, 0.1, 0.9)
        avg_peak = get_avg_between_percentiles(all_epoch_peaks, 0.1, 0.9)
        return avg_time, avg_peak, final_quality
    else:
        return None, None, None


def main(rank, size, args):
    device = torch.device("cpu")
    if args is not None:
        device = torch.device(f"{args.device}:{rank}")
    
    print(f"Rank {rank}, device: {device}")

    # To ensure that the model is initialized the same way on all processes.
    seed_everything(0)
    model = Net(args=args)
    model.to(device)
    if args.implementation == "torch":
        model_ddp = nn.parallel.DistributedDataParallel(model)
    else:
        model_ddp = model
    
    seed_everything(0 + rank)

    train_loader, test_loader = get_loaders(rank, size, args, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    avg_time, avg_peak, final_quality = run_experiment(
        model_ddp, 
        train_loader, 
        test_loader, 
        optimizer, 
        args, 
        device, 
        rank, 
        num_epochs=args.num_epochs
    )

    if rank == 0:
        print(f"Avg Time = {avg_time:.2f}s, "
              f"Avg Peak = {avg_peak/1e6:.2f} MB, "
              f"Final Quality = {final_quality:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="gloo", 
                        choices=["gloo", "nccl"])
    parser.add_argument("--device", type=str, default="cpu", 
                        choices=["cpu", "cuda"])
    parser.add_argument("--implementation", type=str, default="custom", 
                        choices=["custom", "torch"])
    parser.add_argument("--grad_accumulation", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)

    args = parser.parse_args()
    print(args)

    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=partial(main, args=args), backend=args.backend)

"""
torchrun --nproc_per_node=2 --master_port=1234 ddp_cifar100.py \
    --backend=nccl \
    --device=cuda \
    --implementation=torch \
    --grad_accumulation=2 \
    --batch_size=32 \
    --num_epochs=20
"""

"""
torchrun --nproc_per_node=2 --master_port=12345 ddp_cifar100.py \
    --backend=nccl \
    --device=cuda \
    --implementation=custom \
    --grad_accumulation=2 \
    --batch_size=32 \
    --num_epochs=2
"""