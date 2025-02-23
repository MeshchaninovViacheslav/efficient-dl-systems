import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_var, eps: float, momentum: float):
        N, C = input.size(0), input.size(1)

        # Compute local sums along the batch dimension.
        local_sum = input.sum(dim=0)           
        local_sum_sq = (input ** 2).sum(dim=0)

        # Pack the local statistics and count into a single tensor.
        # Note: count is stored as a 1-element tensor.
        count_tensor = torch.tensor([float(N)], device=input.device)
        stats = torch.cat([local_sum, local_sum_sq, count_tensor])
        
        # Aggregate statistics from all processes using a single all-reduce call.
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        # Unpack the aggregated statistics.
        global_sum = stats[:C]
        global_sum_sq = stats[C:2 * C]
        global_count = stats[2 * C].item()  # Total number of examples across processes

        # Compute global mean and variance.
        global_mean = global_sum / global_count
        global_var = global_sum_sq / global_count - global_mean ** 2
        global_std = torch.sqrt(global_var + eps)

        # Normalize the input using the aggregated statistics.
        normalized = (input - global_mean) / global_std

        # Update running statistics.
        running_mean.data = running_mean.data * (1 - momentum) + global_mean * momentum
        running_var.data = running_var.data * (1 - momentum) + global_var * momentum

        # Save context for backward: input, global_mean, global_std, normalized.
        ctx.save_for_backward(input, global_mean, global_std, normalized)
        ctx.global_count = global_count
        ctx.eps = eps

        return normalized

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors and global count.
        input, global_mean, global_std, normalized = ctx.saved_tensors
        N = ctx.global_count
        C = input.size(1)

        # Compute the local sums for gradient statistics.
        grad_sum = grad_output.sum(dim=0)                       
        grad_mul = (grad_output * (input - global_mean)).sum(dim=0)

        grad_stats = torch.cat([grad_sum, grad_mul])
        dist.all_reduce(grad_stats)
        global_grad_sum = grad_stats[:C]
        global_grad_mul = grad_stats[C:2 * C]

        # Compute gradient with respect to the input using the batch norm backward formula.
        # Note that: normalized = (input - global_mean) / global_std.
        # Hence, the gradient dL/dx is given by:
        #   (1/global_std) * [grad_output - (global_grad_sum / N) - normalized * (global_grad_mul / (global_std * N))]
        grad_input = (grad_output - (global_grad_sum / N)
                      - normalized * (global_grad_mul / (global_std * N))) / global_std

        return grad_input, None, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return (input - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return sync_batch_norm.apply(input, self.running_mean, 
                                     self.running_var, self.eps, self.momentum)
