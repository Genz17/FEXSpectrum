import torch
import copy
from torchquad import MonteCarlo, set_up_backend
from L2Norm import L2Norm
from DataGen import DataGen
from Equation import LHS_pde,RHS_pde,true_solution,LaplaceOperator
from Coeff import Coeff,Psi,Coeff_r,Coeff_All
set_up_backend("torch", data_type="float32")
mc = MonteCarlo()
def TreeTrain(f, model, batchOperations, domain, T, dim, order):

    batchSize = model.batchSize
    dataBatch = 100
    treeBuffer = []

    for batch in range(batchSize):
        for i in range(model.treeNum):
            model.treeDict[str(i)].PlaceOP(batchOperations[i][batch])
            model.treeDict[str(i)].LinearGen()

        optimizer = torch.optim.Adam(model.treeDict.parameters())

        for _ in range(20):
            optimizer.zero_grad()
            funcList = [lambda x:sum([Coeff(j,n+1,T,'a',1)*model.treeDict[str(n)](x)+Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda\
                        s:model.treeDict[str(n)](s),x,dim) for n in range(model.treeNum)]) - mc.integrate(lambda \
                        t:f(x,t),1,integration_domain=[[0,T]]) for j in range(1, model.treeNum)]

            loss = sum([Coeff_r(model.treeNum,i+1)*mc.integrate(lambda x:funcList[i](x)**2,dim,1000,domain) for i in range(len(funcList))])
            loss.backward()
            optimizer.step()


        optimizer = torch.optim.LBFGS(model.treeDict.parameters(), lr=1, max_iter=20)

        def closure():
            optimizer.zero_grad()
            funcList = [lambda x:sum([Coeff(j,n+1,T,'a',1)*model.treeDict[str(n)](x)+Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda\
                        s:model.treeDict[str(n)](s),x,dim) for n in range(model.treeNum)]) - mc.integrate(lambda \
                        t:f(x,t),1,integration_domain=[[0,T]]) for j in range(1, model.treeNum)]

            loss = sum([Coeff_r(model.treeNum,i+1)*mc.integrate(lambda x:funcList[i](x)**2,dim,1000,domain) for i in range(len(funcList))])
            loss.backward()
            return loss

        optimizer.step(closure)
        treeBuffer.append(copy.deepcopy(model.treeDict))


    return treeBuffer
