import os
import itertools
import torch
import torch.distributed as dist
import torch.nn as nn
import time
from syncbn import SyncBatchNorm as CustomSyncBatchNorm  # our implementation
# Note: standard PyTorch version will be imported from torch.nn.SyncBatchNorm

def benchmark_syncbn(impl, hid_dim, batch_size, num_iters=50):
    """
    Benchmark one configuration of SyncBatchNorm.

    Args:
      impl (str): either "custom" (our implementation) or "standard" (torch.nn.SyncBatchNorm).
      hid_dim (int): number of features.
      batch_size (int): batch size per process.
      num_iters (int): number of iterations to time (after a warmup phase).

    Returns:
      avg_time_ms (float): average time (ms) per iteration.
      peak_memory_mb (float): peak GPU memory (MB) allocated during the run.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    # Reset peak memory stats for accurate measurement
    torch.cuda.reset_peak_memory_stats(device)
    
    # Set up the BN layer (without affine parameters to match our custom implementation)
    if impl == "custom":
        bn_layer = CustomSyncBatchNorm(hid_dim).to(device)
    elif impl == "standard":
        bn_layer = nn.SyncBatchNorm(hid_dim, affine=False).to(device)
    else:
        raise ValueError("impl must be either 'custom' or 'standard'.")
    bn_layer.train()
    
    # Warmup few iterations to avoid one-time GPU overheads
    for _ in range(5):
        x = torch.randn(batch_size, hid_dim, device=device, requires_grad=True)
        out = bn_layer(x)
        loss = out.sum()
        loss.backward()
    
    # Synchronize before launching the timed runs.
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_iters):
        # Create new input each iteration
        x = torch.randn(batch_size, hid_dim, device=device, requires_grad=True)
        out = bn_layer(x)
        # Compute loss only on a subset if desired; here we simply sum over all output.
        loss = out.sum()
        loss.backward()
    end_event.record()
    # Wait for all work on the GPU to finish.
    torch.cuda.synchronize(device)
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / num_iters

    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    return avg_time_ms, peak_memory_mb

def run_benchmarks():
    """
    Run benchmarks for both implementations over 8 parameter combinations.
    
    Configurations: 
      hid_dims: [128, 256, 512, 1024]
      batch_sizes: [32, 64]
      
    For each configuration and each implementation ('custom' and 'standard'),
    we record the average per–iteration GPU time and the peak memory usage.
    Results are reduced using a max operation so that rank 0 prints the worst-case numbers.
    """
    hid_dims = [128, 256, 512, 1024]
    batch_sizes = [32, 64]
    num_iters = 50
    results = {}  # structure: results[impl][(hid_dim, batch_size)] = (avg_time_ms, peak_memory_mb)
    
    for impl in ["custom", "standard"]:
        results[impl] = {}
        for hid_dim, batch_size in itertools.product(hid_dims, batch_sizes):
            avg_time, peak_mem = benchmark_syncbn(impl, hid_dim, batch_size, num_iters)
            # Create tensors so we can reduce across processes.
            device = torch.device(f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}')
            avg_time_tensor = torch.tensor(avg_time, device=device)
            peak_mem_tensor = torch.tensor(peak_mem, device=device)
            # Reduce max to capture worst-case performance across processes.
            dist.reduce(avg_time_tensor, dst=0, op=dist.ReduceOp.MAX)
            dist.reduce(peak_mem_tensor, dst=0, op=dist.ReduceOp.MAX)
            if dist.get_rank() == 0:
                results[impl][(hid_dim, batch_size)] = (avg_time_tensor.item(), peak_mem_tensor.item())
                print(f"[{impl}] hid_dim: {hid_dim}, batch_size: {batch_size} -> "
                      f"Avg time: {avg_time_tensor.item():.3f} ms, "
                      f"Peak memory: {peak_mem_tensor.item():.2f} MB")
    return results

def main():
    """
    Main entry point for the benchmark.
    
    Initializes the distributed process group (expects environment variables set by torchrun)
    then runs the defined benchmarks.
    """
    # Initialize the process group.
    # We use the NCCL backend for GPU communication.
    dist.init_process_group(backend='nccl')
    run_benchmarks()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()