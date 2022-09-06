import torch

def L2Norm(dim, domain, f):
    left = domain[0]
    right = domain[1]
    x = torch.rand((1000, dim), device='cuda:0', requires_grad=True)
    x = x*(right-left)+left
    y = (f(x)).view(1000)
    return torch.norm(y, 2)
