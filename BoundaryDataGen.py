import torch

def bdDataGen(dim, T, num=100):
    data = 2*(torch.rand((num*dim,dim), device='cuda:0', dtype=torch.float64)-0.5)
    dataT = T*torch.rand((num*dim,1), device='cuda:0', dtype=torch.float64)
    for ii in range(dim):
        data[ii*num:(ii+1)*num,ii] = 0
    return data, dataT
