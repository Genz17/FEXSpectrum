import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
from torchquad import MonteCarlo, set_up_backend, Trapezoid
from Equation import LaplaceOperator,LaplaceOperatorWitht,Diffx,Partialt
from Coeff import Coeff,Psi,Coeff_r,Coeff_All,Phi
from integration1D import integration1DforT
from funcCoeffList import funcCoeffListGen
from outputFunc import outputFunc
import random
torch.set_default_tensor_type('torch.cuda.DoubleTensor')
set_up_backend("torch", data_type="float64")
mc = MonteCarlo()
tp = Trapezoid()
def TreeTrain(f, model, batchOperations, domain, T, dim, order, real_func):

    domainT = copy.deepcopy(domain)
    domainT.append([0,T])
    print(dim)
    print(domain)
    print(domainT)
    batchSize = model.batchSize
    treeBuffer = []
    #X = torch.rand((100,dim), device='cuda:0').view(100,dim)

    for batch in range(batchSize):
        for i in range(model.treeNum):
            model.treeDict[str(i)].PlaceOP(batchOperations[i][batch])
            model.treeDict[str(i)].LinearGen()

        optimizer = torch.optim.Adam(model.treeDict.parameters(), lr=5e-1)
        for _ in range(10):
            optimizer.zero_grad()
            loss = 0
            for j in range(1, model.treeNum+1):
                func = lambda x:sum([Coeff(j,n,T,'a',1)*model.treeDict[str(n-1)](x) - Coeff(j,n,T,'b',1)*LaplaceOperator(lambda \
                            s:model.treeDict[str(n-1)](s),x) for n in range(1, model.treeNum+1)])
                tempLoss = tp.integrate(lambda x:((func(x))**2),dim,5000,domain) - \
                        2*tp.integrate(lambda xt:(func(xt[:,:-1])*(f(xt)*(Psi(order,j,T)(xt[:,-1])).view(-1,1))),dim+1,5000,domainT)
                loss = loss + Coeff_r(model.treeNum,j)*tempLoss
            loss = loss + model.treeNum**(-4)*mc.integrate(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,dim,5000,domain)

            print(_,loss)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                X = 2*(torch.rand((10000,dim), device='cuda:0')-0.5)
                tTest = torch.rand((10000,1), device='cuda:0')
                XT = torch.zeros((10000,dim+1),device='cuda:0')
                XT[:,:-1] = X
                XT[:,-1] = tTest.view(10000)
                z = outputFunc(model,X,tTest,order,T).view(X.shape[0],1)
                y = real_func(XT).view(X.shape[0],1)
                a = torch.sum((z-y)**2,0).view(1)
                b = torch.sum(y**2,0).view(1)
                relerr = torch.sqrt(a/b)
                print('relerr: {}'.format(relerr))


        optimizer = torch.optim.LBFGS(model.treeDict.parameters(), lr=1, max_iter=100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 80, eta_min=0.01, last_epoch=-1, verbose=False)

        def closure():
            optimizer.zero_grad()
            loss = 0
            for j in range(1, model.treeNum+1):
                func = lambda x:sum([Coeff(j,n,T,'a',1)*model.treeDict[str(n-1)](x) - Coeff(j,n,T,'b',1)*LaplaceOperator(lambda \
                            s:model.treeDict[str(n-1)](s),x) for n in range(1, model.treeNum+1)])
                tempLoss = mc.integrate(lambda x:((func(x))**2),dim,5000,domain) - \
                        2*mc.integrate(lambda xt:(func(xt[:,:-1])*(f(xt)*(Psi(order,j,T)(xt[:,-1])).view(-1,1))),dim+1,5000,domainT)
                loss = loss + Coeff_r(model.treeNum,j)*tempLoss
            loss = loss + model.treeNum**(-4)*mc.integrate(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,dim,5000,domain)

            print(loss)
            loss.backward()
            #scheduler.step()
            with torch.no_grad():
                X = 2*(torch.rand((1000,dim), device='cuda:0')-0.5)
                tTest = torch.rand((1000,1), device='cuda:0')
                XT = torch.zeros((1000,dim+1),device='cuda:0')
                XT[:,:-1] = X
                XT[:,-1] = tTest.view(1000)
                z = outputFunc(model,X,tTest,order,T).view(X.shape[0],1)
                y = real_func(XT).view(X.shape[0],1)
                a = torch.sum((z-y)**2,0).view(1)
                b = torch.sum(y**2,0).view(1)
                relerr = torch.sqrt(a/b)
                print('relerr: {}'.format(relerr))
            return loss

        optimizer.step(closure)
        if ((not loss < 1) and (not loss >= 1)):
            for i in range(model.treeNum):
                model.treeDict[str(i)].LinearGen()
            treeBuffer.append(copy.deepcopy(model.treeDict))
        else:
            treeBuffer.append(copy.deepcopy(model.treeDict))


    return treeBuffer

