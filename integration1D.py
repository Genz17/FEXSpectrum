import torch
from Coeff import *
from Equation import Diffx
from torchquad import MonteCarlo, set_up_backend, Trapezoid

def integration1D(func, domain):
    nodes = torch.linspace(domain[0], domain[1], 1000, dtype=torch.float64, device='cuda:0').view(-1,1)
    nodesValue = func(nodes)
    sumValue = 2*(torch.sum(nodesValue,0)) - nodesValue[0,0] - nodesValue[-1,0]
    sumValue = sumValue/2*((domain[1]-domain[0])/(1000-1))
    return sumValue

def integration1DforT(func_x_t, domain, x):
    nodes = torch.linspace(domain[0], domain[1], 1000, dtype=torch.float64, device='cuda:0').view(1,-1)
    xNum = x.shape[0]
    nodeMat = torch.zeros(xNum,1000,dtype=torch.float64,device='cuda:0')
    for j in range(xNum):
        nodeMat[j,:] = func_x_t(x[j,:].view(1,-1),nodes).view(1,-1)
    sumValue = (2*(torch.sum(nodeMat,1)) - nodeMat[:,0] - nodeMat[:,-1])/2*((domain[1]-domain[0])/(999))
    return sumValue.view(-1,1)

