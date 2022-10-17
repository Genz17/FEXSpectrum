import torch
from Equation import Diffx
from torchquad import MonteCarlo, set_up_backend, Boole
torch.set_default_tensor_type('torch.cuda.DoubleTensor')
set_up_backend("torch", data_type="float64")
integrator = Boole()
N = 1000
Vec = torch.ones(N, 1, device='cuda:0')

def integration1D(func, upLim):
    denseMat = torch.zeros((upLim.shape[0], 1000), device='cuda:0')
    for i in range(1000):
        denseMat[:, i:i+1] = upLim - 1e-4*(999-i)
    nodeMat = torch.trapezoid(denseMat).view(-1,1)
    return nodeMat

