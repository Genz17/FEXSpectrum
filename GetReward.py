import torch
import copy
from DataGen import DataGen
from Equation import LHS_pde,RHS_pde,true_solution,LaplaceOperator
from Coeff import Coeff

def GetReward(model, batchOperations, domainLeft, domainRight, T, dim, order):

    batchSize = model.batchSize
    treeBuffer = []

    for batch in range(batchSize):
        for treeinx in range(model.treeNum):
            batchOperation = batchOperations[treeinx][batch]
            model.treeDict[str(treeinx)].PlaceOP(batchOperation)
            model.treeDict[str(treeinx)].LinearGen()

        optimizer = torch.optim.Adam(model.treeDict.parameters())

        for _ in range(20):
            data = torch.rand((100, dim), device='cuda:0', requires_grad=True)*2-1
            optimizer.zero_grad()
            loss = 0
            for j in range(model.treeNum):
                ans = 0
                for treeinx in range(model.treeNum):
                    res = model.treeDict[str(treeinx)](data)
                    ans += Coeff(j, treeinx, T, 'a', order)*res
                    ans += Coeff(j, treeinx, T, 'b', order)*LaplaceOperator(res, data, dim)
                loss += torch.norm(ans)
            loss.backward()
            optimizer.step()


        optimizer = torch.optim.LBFGS(model.treeDict.parameters(), lr=1, max_iter=20)

        def closure():
            data = torch.rand((100, dim), device='cuda:0', requires_grad=True)*2-1
            optimizer.zero_grad()
            loss = 0
            for j in range(model.treeNum):
                ans = 0
                for treeinx in range(model.treeNum):
                    res = model.treeDict[str(treeinx)](data)
                    ans += Coeff(j, treeinx, T, 'a', order)*res
                    ans += Coeff(j, treeinx, T, 'b', order)*LaplaceOperator(res, data, dim)
                loss += torch.norm(ans)
            loss.backward()
            return loss

        optimizer.step(closure)
        treeBuffer.append(copy.deepcopy(model.treeDict))


    return treeBuffer
