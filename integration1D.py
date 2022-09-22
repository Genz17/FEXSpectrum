import torch

def integration1D(func, domain):
    nodes = torch.linspace(domain[0], domain[1], 1000).view(-1,1)
    nodesValue = func(nodes)
    sumValue = 2*(torch.sum(nodesValue,0)) - nodesValue[0,0] - nodesValue[-1,0]
    sumValue = sumValue/2*((domain[1]-domain[0])/(1000-1))
    return sumValue

