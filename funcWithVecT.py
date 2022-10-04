import torch

def funcTrans(func,vecX,vecT):
    tNum = vecT.shape[0]
    res = torch.zeros(tNum,device='cuda:0',dtype=torch.float64)
    for tt in range(tNum):
        res[tt] = func(vecX[tt,:], vecT[tt]).view(1)

    return res.view(-1,1)

