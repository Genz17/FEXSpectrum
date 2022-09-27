import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from torchquad import MonteCarlo, set_up_backend, Trapezoid
from Equation import LHS_pde,RHS_pde,true_solution,LaplaceOperator,Diffx,Partialt
from Coeff import Coeff,Psi,Coeff_r,Coeff_All,Phi
from integration1D import integration1DforT,integration1D
set_up_backend("torch", data_type="float64")
mc = MonteCarlo()
tp = Trapezoid()
def TreeTrain(f, model, batchOperations, domain, T, dim, order, real_func):

    batchSize = model.batchSize
    treeBuffer = []
    x = torch.linspace(0,1,1000, device='cuda:0').view(1000,1)

    for batch in range(batchSize):
        for i in range(model.treeNum):
            model.treeDict[str(i)].PlaceOP(batchOperations[i][batch])
            model.treeDict[str(i)].LinearGen()

        optimizer = torch.optim.Adam(model.treeDict.parameters(), lr=1e-2)

        for _ in range(10):
            optimizer.zero_grad()
            funcList = [lambda x:(sum([Coeff(j,n+1,T,'a',1)*model.treeDict[str(n)](x) - Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda \
                        s:model.treeDict[str(n)](s),x) for n in range(model.treeNum)]) - integration1DforT(
                            lambda s,l:f(s,l)*Psi(order, j, T)(l),T,x)) for j in range(1, model.treeNum+3)]
            lossList = [Coeff_r(model.treeNum,i+1)*mc.integrate(lambda x:(funcList[i](x))**2,1,1000,domain) for i in range(len(funcList))]
            loss = sum(lossList) + 0.1*model.treeNum**(-4)*mc.integrate(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,1,1000,domain)
            print(_,loss)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                outputFunc = lambda x,t: sum([model.treeDict[str(j)](x)*Phi(order,j+1,T)(t) for j in range(model.treeNum)])
                z = outputFunc(x, 0.1).view(1000,1)
                y = real_func(x,torch.tensor(0.1)).view(1000,1)
                print('relerr: {}'.format(torch.norm(y-z)/torch.norm(y)))
                z = outputFunc(x, 0.5).view(1000,1)
                y = real_func(x,torch.tensor(0.5)).view(1000,1)
                print('relerr: {}'.format(torch.norm(y-z)/torch.norm(y)))
                z = outputFunc(x, 0.9).view(1000,1)
                y = real_func(x,torch.tensor(0.9)).view(1000,1)
                print('relerr: {}'.format(torch.norm(y-z)/torch.norm(y)))
                del outputFunc
                del z
                del y
            del funcList
            del lossList
            del loss


        optimizer = torch.optim.LBFGS(model.treeDict.parameters(), lr=1, max_iter=20)

        def closure():
            optimizer.zero_grad()
            funcList = [lambda x:(sum([Coeff(j,n+1,T,'a',1)*model.treeDict[str(n)](x) - Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda \
                        s:model.treeDict[str(n)](s),x) for n in range(model.treeNum)]) - integration1DforT(
                            lambda s,l:f(s,l)*Psi(order, j, T)(l),T,x)) for j in range(1, model.treeNum+3)]
            lossList = [Coeff_r(model.treeNum,i+1)*mc.integrate(lambda x:(funcList[i](x))**2,dim,1000,domain) for i in range(len(funcList))]
            loss = sum(lossList) + 0.1*model.treeNum**(-4)*mc.integrate(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,dim,1000,domain)
            print(loss)
            loss.backward()
            with torch.no_grad():
                outputFunc = lambda x,t: sum([model.treeDict[str(j)](x)*Phi(order,j+1,T)(t) for j in range(model.treeNum)])
                x = torch.linspace(0,1,1000, device='cuda:0').view(1000,1)
                z1 = outputFunc(x, 0.1).view(1000,1)
                y1 = real_func(x,torch.tensor(0.1)).view(1000,1)
                print('relerr: {}'.format(torch.norm(y1-z1)/torch.norm(y1)))
                z2 = outputFunc(x, 0.5).view(1000,1)
                y2 = real_func(x,torch.tensor(0.5)).view(1000,1)
                print('relerr: {}'.format(torch.norm(y2-z2)/torch.norm(y2)))
                z3 = outputFunc(x, 0.9).view(1000,1)
                y3 = real_func(x,torch.tensor(0.9)).view(1000,1)
                print('relerr: {}'.format(torch.norm(y3-z3)/torch.norm(y3)))
            return loss

        optimizer.step(closure)
        treeBuffer.append(copy.deepcopy(model.treeDict))


    return treeBuffer
