import torch
from Equation import Diffx
from torchquad import MonteCarlo, set_up_backend, Trapezoid
set_up_backend("torch", data_type="float64")
mc = MonteCarlo()
tp = Trapezoid()

def integration1DforT(func_x_t, T, x):
    xNum = x.shape[0]
    tList = torch.linspace(0,1,1000, device='cuda:0').view(1,-1)
    nodeMat = torch.zeros(xNum,1,device='cuda:0')
    for j in range(xNum):
        func = lambda t:func_x_t(x[j,:].view(1,-1),t)
        nodeMat[j,:] = tp.integrate(func, 1, 1000, [[0,T]])
    return nodeMat

