import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import pytest
from syncbn import SyncBatchNorm
from functools import partial
import random


def init_process(rank, size, fn, master_port, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def worker_process(rank, world_size, hid_dim, batch_size, queue):
    """Worker process function that runs SyncBN."""
    torch.manual_seed(42 + rank)
    inputs = torch.randn(batch_size, hid_dim)
    inputs.requires_grad = True

    sync_bn = SyncBatchNorm(hid_dim)
    outputs = sync_bn(inputs)
    loss = outputs[:batch_size//2].sum()
    loss.backward()
    
    queue.put({
        'rank': rank,
        'outputs': outputs.detach().numpy(),
        'grad_inputs': inputs.grad.detach().numpy(),
    })


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("hid_dim", [128, 256, 512, 1024])
@pytest.mark.parametrize("batch_size", [32, 64])
def test_batchnorm(num_workers, hid_dim, batch_size):
    # Set up multiprocessing context
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    
    # Launch worker processes
    port = random.randint(25000, 30000)
    processes = []
    for rank in range(num_workers):
        p = ctx.Process(
            target=init_process,
            args=(rank, num_workers, 
                  partial(
                        worker_process, 
                        hid_dim=hid_dim, 
                        batch_size=batch_size, 
                        queue=queue
                        ), 
                  port)
        )
        p.start()
        processes.append(p)

    # Create regular BatchNorm for comparison
    inputs_full = torch.randn(batch_size * num_workers, hid_dim)
    for i in range(num_workers):
        torch.manual_seed(42 + i)
        inputs_full[i * batch_size:(i + 1) * batch_size] = torch.randn(batch_size, hid_dim)
    inputs_full.requires_grad = True
    
    bn = nn.BatchNorm1d(hid_dim, affine=False)

    # Forward pass with regular BatchNorm
    outputs_bn = bn(inputs_full)

    # Compute loss (sum over first B/2 samples for each worker)
    loss_bn = torch.tensor(0.)
    for i in range(num_workers):
        start_idx = i * batch_size
        mid_idx = start_idx + batch_size // 2
        loss_bn += outputs_bn[start_idx:mid_idx].sum()
    
    # Backward pass
    loss_bn.backward()

    worker_results = [queue.get() for _ in range(num_workers)]
    for p in processes:
        p.join()

    # Compare outputs and gradients
    atol = 1e-3
    rtol = 0.0

    # Compare each worker's outputs and gradients against the corresponding slices
    worker_results = sorted(worker_results, key=lambda x: x['rank'])
    for res in worker_results:
        r = res['rank']
        worker_out = torch.from_numpy(res['outputs'])
        worker_grad = torch.from_numpy(res['grad_inputs'])
        ref_out = outputs_bn[r * batch_size:(r + 1) * batch_size]
        ref_grad = inputs_full.grad[r * batch_size:(r + 1) * batch_size]
        assert torch.allclose(worker_out, ref_out, atol=atol, rtol=rtol), \
            f"Rank {r} outputs don't match: max diff = " \
            f"{(worker_out - ref_out).abs().max()}"
        assert torch.allclose(worker_grad, ref_grad, atol=atol, rtol=rtol), \
            f"Rank {r} gradients don't match: max diff = " \
            f"{(worker_grad - ref_grad).abs().max()}"
