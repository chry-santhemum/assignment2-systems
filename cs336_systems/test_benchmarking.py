from .benchmarking import benchmark_basic, benchmark_attn
from cs336_basics.model import (
    BasicsTransformerLM, 
    scaled_dot_product_attention,
)
import torch


def test_benchmark_basic():
    # args setting
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=128,
        d_model=16,
        num_layers=2,
        num_heads=1,
        d_ff=64,
        rope_theta=10000.0,
    )
    batch_size = 8
    seq_len = 128
    num_warmup = 5
    num_samples = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning benchmark on {device}")

    stats_both = benchmark_basic(model, batch_size, seq_len, num_warmup, num_samples, backward=True, compiled=False, device=device)
    stats_fwd = benchmark_basic(model, batch_size, seq_len, num_warmup, num_samples, backward=False, compiled=False, device=device)
    stats_jit_both = benchmark_basic(model, batch_size, seq_len, num_warmup, num_samples, backward=True, compiled=True, device=device)
    stats_jit_fwd = benchmark_basic(model, batch_size, seq_len, num_warmup, num_samples, backward=False, compiled=True, device=device)

    print(stats_both)
    print(stats_fwd)
    print(stats_jit_both)
    print(stats_jit_fwd)

    assert stats_both["mean"] > stats_fwd["mean"]
    assert stats_jit_both["mean"] > stats_jit_fwd["mean"]
    assert stats_both["mean"] > stats_jit_both["mean"]
    assert stats_fwd["mean"] > stats_jit_fwd["mean"]


def test_benchmark_attn():
    # args setting
    d_head = 8
    seq_len = 128
    num_warmup = 5
    num_samples = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning benchmark on {device}")

    stats = benchmark_attn(d_head, seq_len, num_warmup, num_samples, compiled=False, device=device)
    stats_jit = benchmark_attn(d_head, seq_len, num_warmup, num_samples, compiled=True, device=device)
    print(stats)
    print(stats_jit)

    assert stats["forward_mean"] > stats_jit["forward_mean"]
    assert stats["backward_mean"] > stats_jit["backward_mean"]



