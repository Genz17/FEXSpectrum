import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
from torchquad import MonteCarlo, set_up_backend, Boole
from Equation import LaplaceOperator,LaplaceOperatorWitht,Diffx,Partialt
from Coeff import Coeff,Psi,Coeff_r,Coeff_All,Phi
from funcCoeffList import funcCoeffListGen
from outputFunc import intFunc,outputFunc
import random
torch.set_default_tensor_type('torch.cuda.DoubleTensor')
set_up_backend("torch", data_type="float64")
mc = MonteCarlo()
bl = Boole()
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
        model.tree.PlaceOP(batchOperations[batch])
        model.tree.LinearGen()
        model.tree.OperationsRefresh()

        optimizer = torch.optim.Adam(model.tree.parameters(), lr=5e-3)
        for _ in range(10):
            optimizer.zero_grad()
            loss = 0
            for j in range(1, model.outNum+1):
                tempLoss = bl.integrate(lambda x:((intFunc(model.tree, x, j, T, order))**2),dim,5000,domain) - \
                        bl.integrate(lambda xt:2*(intFunc(model.tree, xt[:,:-1], j, T, order)*(f(xt)*(Psi(order,j,T)(xt[:,-1])).view(-1,1))),dim+1,5000,domainT)
                loss = loss + Coeff_r(model.outNum,j)*tempLoss
            #loss = loss + model.treeNum**(-4)*mc.integrate(lambda x:(LaplaceOperator(lambda s:model.tree[str(model.treeNum-1)](s),x))**2,dim,5000,domain)

            print(_,loss)
            loss.backward()
            optimizer.step()
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


        optimizer = torch.optim.LBFGS(model.tree.parameters(), lr=0.5, max_iter=80)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 400, eta_min=0.01, last_epoch=-1, verbose=False)

        def closure():
            optimizer.zero_grad()
            loss = 0
            for j in range(1, model.outNum+1):
                tempLoss = bl.integrate(lambda x:((intFunc(model.tree, x, j, T, order))**2),dim,5000,domain) - \
                        bl.integrate(lambda xt:2*(intFunc(model.tree, xt[:,:-1], j, T, order)*(f(xt)*(Psi(order,j,T)(xt[:,-1])).view(-1,1))),dim+1,5000,domainT)
                loss = loss + Coeff_r(model.outNum,j)*tempLoss
            #loss = loss + model.treeNum**(-4)*mc.integrate(lambda x:(LaplaceOperator(lambda s:model.tree[str(model.treeNum-1)](s),x))**2,dim,5000,domain)

            print(loss, type(loss))
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
        if ((not relerr < 1) and (not relerr >= 1)) or relerr > 1e3:
            model.tree.LinearGen()
            model.tree.OperationsRefresh()
            print('Now we are at nan.')
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
            treeBuffer.append(copy.deepcopy(model.tree))
        else:
            print("Now this choices are fine.")
            treeBuffer.append(copy.deepcopy(model.tree))


    return treeBuffer

