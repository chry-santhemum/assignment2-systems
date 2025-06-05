from matrix_vector import WeightedSumFunc
import torch

f_weighted_sum = WeightedSumFunc.apply

def test_weighted_sum_fwd():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("CUDA is not available, test is skipped")
        return

    x = torch.randn(128, 10, device=device)
    weight = torch.randn(10, device=device)

    y = f_weighted_sum(x, weight)
    print(y)


def test_weighted_sum_both():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("CUDA is not available, test is skipped")
        return

    x = torch.randn(128, 10, device=device, requires_grad=True)
    weight = torch.randn(10, device=device, requires_grad=True)

    y = f_weighted_sum(x, weight)
    print(y)
    loss = y.sum()
    loss.backward()

    print(x.grad)
    print(weight.grad)