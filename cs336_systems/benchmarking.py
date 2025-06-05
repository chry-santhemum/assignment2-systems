from cs336_basics.model import (
    BasicsTransformerLM, 
    scaled_dot_product_attention,
)

import numpy as np
from timeit import default_timer as timer
import torch
import torch.nn.functional as F


def benchmark_basic(
    model: BasicsTransformerLM, 
    batch_size: int, 
    seq_len: int, 
    num_warmup: int = 1,
    num_samples: int = 5,
    backward: bool = False,  # both forward and backward
    compiled: bool = False,
    device: str = "cuda"
) -> dict[str, float]:

    # Move model to device
    model = model.to(device)

    if compiled:
        model = torch.compile(model)
    
    # generate random data
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
    label_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

    def do_one_pass(backward: bool = False):
        if backward:
            model.zero_grad()
            output_ten = model(input_ids)
            # Reshape for cross_entropy: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
            loss = F.cross_entropy(output_ten.view(-1, output_ten.size(-1)), label_ids.view(-1))
            loss.backward()
        else:
            with torch.no_grad():
                model(input_ids)


    for _ in range(num_warmup):
        do_one_pass(backward)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(num_samples):
        start = timer()
        do_one_pass(backward)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(timer() - start)
    
    stats = {
        "mean": np.mean(times).item(),
        "std": np.std(times).item(),
    }
    return stats


def benchmark_attn(
    d_head: int,
    seq_len: int, 
    num_warmup: int = 5,
    num_samples: int = 100,
    compiled: bool = False,
    device: str = "cuda"
) -> dict[str, float]:
    Q = torch.randn(8, seq_len, d_head, device=device)
    K = torch.randn(8, seq_len, d_head, device=device)
    V = torch.randn(8, seq_len, d_head, device=device)

    if compiled:
        sdpa = torch.compile(scaled_dot_product_attention)
    else:
        sdpa = scaled_dot_product_attention

    for _ in range(num_warmup):
        sdpa(Q, K, V)
    
    if device == "cuda":
        torch.cuda.synchronize()

    forward_times = []
    for _ in range(num_samples):
        start = timer()
        sdpa(Q, K, V)
        if device == "cuda":
            torch.cuda.synchronize()
        forward_times.append(timer() - start)
    
    # record memory usage
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
    else:
        allocated = 0.
        reserved = 0.

    backward_times = []
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    for _ in range(num_samples):
        loss = sdpa(Q, K, V).sum()
        start = timer()
        if device == "cuda":
            torch.cuda.synchronize()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        backward_times.append(timer() - start)

    stats = {
        "forward_mean": np.mean(forward_times).item(),
        "forward_std": np.std(forward_times).item(),
        "backward_mean": np.mean(backward_times).item(),
        "backward_std": np.std(backward_times).item(),
        "allocated_after_forward_GB": allocated,
        "reserved_after_forward_GB": reserved,
    }
    return stats

