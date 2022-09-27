import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from torchquad import MonteCarlo, set_up_backend, Trapezoid
from Equation import LHS_pde,RHS_pde,true_solution,LaplaceOperator,Diffx,Partialt
from Coeff import Coeff,Psi,Coeff_r,Coeff_All,Phi
from integration1D import integration1DforT,integration1D
from funcCoeffList import funcCoeffListGen
set_up_backend("torch", data_type="float64")
mc = MonteCarlo()
tp = Trapezoid()
def TreeTrain(f, model, batchOperations, domain, T, dim, order, real_func):

    batchSize = model.batchSize
    treeBuffer = []
    X = torch.rand((100,dim), device='cuda:0').view(100,2)

    for batch in range(batchSize):
        for i in range(model.treeNum):
            model.treeDict[str(i)].PlaceOP(batchOperations[i][batch])
            model.treeDict[str(i)].LinearGen()

        optimizer = torch.optim.Adam(model.treeDict.parameters(), lr=1e-2)

        for _ in range(10):
            optimizer.zero_grad()
            loss = 0
            for j in range(1, model.treeNum+3):
                func = lambda x:(sum([Coeff(j,n,T,'a',1)*model.treeDict[str(n-1)](x) - Coeff(j,n,T,'b',1)*LaplaceOperator(lambda \
                            s:model.treeDict[str(n-1)](s),x) for n in funcCoeffListGen(j, model.treeNum,1)]) - integration1DforT(
                                lambda s,l:f(s,l)*Psi(order, j, T)(l),T,x))
                loss = loss + mc.integrate(lambda x:(func(x))**2,dim,500,domain)
            loss = loss + 0.1*model.treeNum**(-4)*mc.integrate(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,dim,500,domain)
            print(_,loss)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                outputFunc = lambda x,t: sum([model.treeDict[str(j)](x)*Phi(order,j+1,T)(t) for j in range(model.treeNum)])
                z = outputFunc(X, 0.1).view(100,1)
                y = real_func(X,torch.tensor(0.1)).view(100,1)
                print('relerr: {}'.format(torch.norm(y-z)/torch.norm(y)))
                z = outputFunc(X, 0.5).view(100,1)
                y = real_func(X,torch.tensor(0.5)).view(100,1)
                print('relerr: {}'.format(torch.norm(y-z)))
                z = outputFunc(X, 0.9).view(100,1)
                y = real_func(X,torch.tensor(0.9)).view(100,1)
                print('relerr: {}'.format(torch.norm(y-z)/torch.norm(y)))
                del outputFunc
                del z
                del y
            del func
            del loss


        optimizer = torch.optim.LBFGS(model.treeDict.parameters(), lr=0.5, max_iter=30)

        def closure():
            optimizer.zero_grad()
            loss = 0
            for j in range(1, model.treeNum+3):
                func = lambda x:(sum([Coeff(j,n,T,'a',1)*model.treeDict[str(n-1)](x) - Coeff(j,n,T,'b',1)*LaplaceOperator(lambda \
                            s:model.treeDict[str(n-1)](s),x) for n in funcCoeffListGen(j, model.treeNum,1)]) - integration1DforT(
                                lambda s,l:f(s,l)*Psi(order, j, T)(l),T,x))
                loss = loss + Coeff_r(model.treeNum,j)*mc.integrate(lambda x:(func(x))**2,dim,1000,domain)
            loss = loss + 0.1*model.treeNum**(-4)*mc.integrate(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,dim,1000,domain)
            print(loss)
            loss.backward()
            with torch.no_grad():
                outputFunc = lambda x,t: sum([model.treeDict[str(j)](x)*Phi(order,j+1,T)(t) for j in range(model.treeNum)])
                z = outputFunc(X, 0.1).view(100,1)
                y = real_func(X,torch.tensor(0.1)).view(100,1)
                print('relerr: {}'.format(torch.norm(y-z)/torch.norm(y)))
                z = outputFunc(X, 0.5).view(100,1)
                y = real_func(X,torch.tensor(0.5)).view(100,1)
                print('relerr: {}'.format(torch.norm(y-z)))
                z = outputFunc(X, 0.9).view(100,1)
                y = real_func(X,torch.tensor(0.9)).view(100,1)
                print('relerr: {}'.format(torch.norm(y-z)/torch.norm(y)))
                del outputFunc
                del z
                del y
            return loss

        optimizer.step(closure)
        treeBuffer.append(copy.deepcopy(model.treeDict))


    return treeBuffer
