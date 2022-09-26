import torch
import numpy as np
import copy
from torchquad import MonteCarlo, set_up_backend, Trapezoid
from Equation import LHS_pde,RHS_pde,true_solution,LaplaceOperator
from Coeff import Coeff,Psi,Coeff_r,Coeff_All
from integration1D import integration1DforT,integration1D
set_up_backend("torch", data_type="float64")
mc = MonteCarlo()
tp = Trapezoid()
def TreeTrain(f, model, batchOperations, domain, T, dim, order):

    batchSize = model.batchSize
    treeBuffer = []

    for batch in range(batchSize):
        for i in range(model.treeNum):
            model.treeDict[str(i)].PlaceOP(batchOperations[i][batch])
            model.treeDict[str(i)].LinearGen()

        optimizer = torch.optim.Adam(model.treeDict.parameters(), lr=1e-2)

        for _ in range(10):
            optimizer.zero_grad()
            funcList = [lambda x:(sum([Coeff(j,n+1,T,'a',1)*model.treeDict[str(n)](x) - Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda \
                        s:model.treeDict[str(n)](s),x) for n in range(model.treeNum)]) - integration1DforT(
                            lambda s,l:f(s,l)*Psi(order, j, T)(l),[0,T],x)) for j in range(1, model.treeNum+1)]

            lossList = [Coeff_r(model.treeNum,i+1)*integration1D(lambda x:(funcList[i](x))**2,[0,1]) for i in range(len(funcList))]
            loss = sum(lossList) + model.treeNum**(-4)*integration1D(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,[0,1])
            print(_,loss)
            loss.backward()
            del funcList
            del lossList
            del loss
            optimizer.step()


        optimizer = torch.optim.LBFGS(model.treeDict.parameters(), lr=0.5, max_iter=20)

        def closure():
            optimizer.zero_grad()
            funcList = [lambda x:(sum([Coeff(j,n+1,T,'a',1)*model.treeDict[str(n)](x[:,:]) - Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda \
                        s:model.treeDict[str(n)](s[:,:]),x) for n in range(model.treeNum)]) - integration1DforT(
                            lambda s,l:f(s,l)*Psi(order, j, T)(l),[0,T],x)) for j in range(1, model.treeNum+1)]

            lossList = [Coeff_r(model.treeNum,i+1)*integration1D(lambda x:(funcList[i](x[:,:]))**2,[0,1]) for i in range(len(funcList))]
            loss = sum(lossList) + model.treeNum**(-4)*integration1D(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,[0,1])
            print(loss)
            loss.backward()
            return loss

        optimizer.step(closure)
        treeBuffer.append(copy.deepcopy(model.treeDict))


    return treeBuffer
