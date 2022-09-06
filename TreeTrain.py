import torch
import copy
from torchquad import MonteCarlo, set_up_backend
from L2Norm import L2Norm
from DataGen import DataGen
from Equation import LHS_pde,RHS_pde,true_solution,LaplaceOperator
from Coeff import Coeff,Psi,Coeff_r
set_up_backend("torch", data_type="float32")
mc = MonteCarlo()
def TreeTrain(f, model, batchOperations, domain, T, dim, order):

    batchSize = model.batchSize
    dataBatch = 100
    treeBuffer = []

    for batch in range(batchSize):
        for treeinx in range(model.treeNum):
            batchOperation = batchOperations[treeinx][batch]
            model.treeDict[str(treeinx)].PlaceOP(batchOperation)
            model.treeDict[str(treeinx)].LinearGen()

        optimizer = torch.optim.Adam(model.treeDict.parameters())

        for _ in range(20):
            optimizer.zero_grad()
            func = lambda x,j: sum([Coeff(j,treeinx,T,'a', order)*model.treeDict[str(treeinx)](x)+Coeff(j,treeinx,T,'b',order)*LaplaceOperator(model.treeDict[str(treeinx)](x),x,dim)\
for treeinx in range(model.treeNum)])
            funcList = [lambda x:func(x,j)-mc.integrate(lambda t:f(x, t)*Psi(order, j, T)(t), dim=1, integration_domain=[[0,T]]) for j in range(1, model.treeNum+1)]
            lossList = [Coeff_r(model.treeNum, j)*L2Norm(dim, domain, funcList[j-1])**2 for j in range(1,model.treeNum+1)]
            loss = sum(lossList) + model.treeNum**(-4)*L2Norm(dim, domain, lambda x:LaplaceOperator(model.treeDict[str(model.treeNum-1)](x),x,dim))
            loss.backward()
            optimizer.step()


        optimizer = torch.optim.LBFGS(model.treeDict.parameters(), lr=1, max_iter=20)

        def closure():
            optimizer.zero_grad()
            func = lambda x,j: sum([Coeff(j,treeinx,T,'a', order)*model.treeDict[str(treeinx)](x)+Coeff(j,treeinx,T,'b',order)*LaplaceOperator(model.treeDict[str(treeinx)](x),x,dim)\
for treeinx in range(model.treeNum)])
            funcList = [lambda x:func(x,j)-mc.integrate(lambda t:f(x, t)*Psi(order, j, T)(t), dim=1, integration_domain=[[0,T]]) for j in range(1, model.treeNum+1)]
            lossList = [Coeff_r(model.treeNum, j)*L2Norm(dim, domain, funcList[j-1])**2 for j in range(1,model.treeNum+1)]
            loss = sum(lossList) + model.treeNum**(-4)*L2Norm(dim, domain, lambda x:LaplaceOperator(model.treeDict[str(model.treeNum-1)](x),x,dim))
            loss.backward()
            return loss

        optimizer.step(closure)
        treeBuffer.append(copy.deepcopy(model.treeDict))


    return treeBuffer
