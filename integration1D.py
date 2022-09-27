import torch
from Equation import Diffx
from torchquad import MonteCarlo, set_up_backend, Trapezoid
set_up_backend("torch", data_type="float64")
mc = MonteCarlo()
tp = Trapezoid()

def integration1D(func, domain):
    nodes = torch.linspace(domain[0], domain[1], 500, dtype=torch.float64, device='cuda:0').view(-1,1)
    nodesValue = func(nodes)
    sumValue = 2*(torch.sum(nodesValue,0)) - nodesValue[0,0] - nodesValue[-1,0]
    sumValue = sumValue/2*((domain[1]-domain[0])/(500-1))
    return sumValue

def integration1DforT(func_x_t, T, x):
    xNum = x.shape[0]
    nodeMat = torch.zeros(xNum,1,dtype=torch.float64,device='cuda:0')
    for j in range(xNum):
        func = lambda t:func_x_t(x[j,:].view(1,-1),t)
        nodeMat[j,:] = tp.integrate(func, 1, 1000, [[0,T]])
    return nodeMat

