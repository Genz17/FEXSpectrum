import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
from torchquad import MonteCarlo, set_up_backend, Trapezoid
from Equation import LaplaceOperator,Diffx,Partialt
from Coeff import Coeff,Psi,Coeff_r,Coeff_All,Phi
from integration1D import integration1DforT
from funcCoeffList import funcCoeffListGen
from outputFunc import outputFunc
import random
set_up_backend("torch", data_type="float64")
mc = MonteCarlo()
tp = Trapezoid()
def TreeTrain(f, model, batchOperations, domain, T, dim, order, real_func):

    domainT = copy.deepcopy(domain)
    domainT.append([0,T])
    print(domain)
    print(domainT)
    batchSize = model.batchSize
    treeBuffer = []
    X = 2*(torch.rand((100,dim), device='cuda:0')-0.5)
    #X = torch.rand((100,dim), device='cuda:0').view(100,dim)

    for batch in range(batchSize):
        for i in range(model.treeNum):
            model.treeDict[str(i)].PlaceOP(batchOperations[i][batch])
            model.treeDict[str(i)].LinearGen()

        optimizer = torch.optim.Adam(model.treeDict.parameters(), lr=1e-2)
        for _ in range(10):
            optimizer.zero_grad()
            loss = 0
            for j in range(1, model.treeNum+1):
                func = lambda x:sum([Coeff(j,n,T,'a',1)*model.treeDict[str(n-1)](x) - Coeff(j,n,T,'b',1)*LaplaceOperator(lambda \
                            s:model.treeDict[str(n-1)](s),x) for n in range(1, model.treeNum+1)])
                tempLoss = mc.integrate(lambda x:((func(x))**2),dim,1000,domain) - \
                        mc.integrate(lambda xt:(func(xt[:,:-1])*(f(xt[:,:-1],xt[:,-1])*(Psi(order,j,T)(xt[:,-1])).view(-1,1))),dim+1,1000,domainT)
                loss = loss + tempLoss
            print(_,loss)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                tTest = torch.rand(1, device='cuda:0')
                z = outputFunc(model,X,tTest,order,T).view(100,1)
                y = real_func(X,tTest).view(100,1)
                print('relerr: {}'.format(torch.norm(y-z)/torch.norm(y)))


        optimizer = torch.optim.LBFGS(model.treeDict.parameters(), lr=0.5, max_iter=5000)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.01, last_epoch=-1, verbose=False)

        def closure():
            optimizer.zero_grad()
            loss = 0
            for j in range(1, model.treeNum+1):
                func = lambda x:sum([Coeff(j,n,T,'a',1)*model.treeDict[str(n-1)](x) - Coeff(j,n,T,'b',1)*LaplaceOperator(lambda \
                            s:model.treeDict[str(n-1)](s),x) for n in range(1, model.treeNum+1)])
                tempLoss = mc.integrate(lambda x:((func(x))**2),dim,1000,domain) - \
                        2*mc.integrate(lambda xt:(func(xt[:,:-1])*(f(xt[:,:-1],xt[:,-1])*(Psi(order,j,T)(xt[:,-1])).view(-1,1))),dim+1,1000,domainT)
                loss = loss + tempLoss
            #loss = loss + 0.1*model.treeNum**(-4)*mc.integrate(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,dim,1000,domain)
            print(loss)
            loss.backward()
            scheduler.step()
            with torch.no_grad():
                tTest = torch.rand(1, device='cuda:0')
                z = outputFunc(model,X,tTest,order,T).view(100,1)
                y = real_func(X,tTest).view(100,1)
                print('relerr: {}'.format(torch.norm(y-z)/torch.norm(y)))
            return loss

        optimizer.step(closure)
        treeBuffer.append(copy.deepcopy(model.treeDict))


    return treeBuffer
