import torch
from Coeff import *
from Equation import Diffx
def integration1D(func, domain):
    nodes = torch.linspace(domain[0], domain[1], 10000, dtype=torch.float64).view(-1,1)
    nodesValue = func(nodes)
    sumValue = 2*(torch.sum(nodesValue,0)) - nodesValue[0,0] - nodesValue[-1,0]
    sumValue = sumValue/2*((domain[1]-domain[0])/(10000-1))
    return sumValue

def integration1DforT(func_x_t, domain, x):
    nodes = torch.linspace(domain[0], domain[1], 500, dtype=torch.float64)
    xNum = x.shape[0]
    nodeMat = torch.zeros(xNum,500,dtype=torch.float64,device='cuda:0')
    for j in range(xNum):
        nodeMat[j,:] = func_x_t(x[j],nodes)
    sumValue = (2*(torch.sum(nodeMat,1)) - nodeMat[:,0] - nodeMat[:,-1])/2*((domain[1]-domain[0])/(499))
    return sumValue.view(-1,1)
