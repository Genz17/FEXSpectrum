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

    for batch in range(batchSize):
        for i in range(model.treeNum):
            model.treeDict[str(i)].PlaceOP(batchOperations[i][batch])
            model.treeDict[str(i)].LinearGen()

        optimizer = torch.optim.Adam(model.treeDict.parameters(), lr=1e-1)

        for _ in range(0):
            optimizer.zero_grad()
            funcList = [lambda x:(sum([integration1D(lambda x:Diffx(Phi(1,n+1,1),x)*Psi(1,j,1)(x), [0,1])*model.treeDict[str(n)](x) - integration1D(lambda x:Phi(1,n+1,1)(x)*Psi(1,j,1)(x),[0,1])*LaplaceOperator(lambda \
                        s:model.treeDict[str(n)](s),x) for n in range(model.treeNum)]) - integration1DforT(
                            lambda s,l:f(s,l)*Psi(order, j, T)(l),[0,T],x)) for j in range(1, model.treeNum+1)]

            #lossList = [Coeff_r(model.treeNum,i+1)*integration1D(lambda x:(funcList[i](x))**2,[0,1]) for i in range(len(funcList))]
            lossList = [integration1D(lambda x:(funcList[i](x))**2,[0,1]) for i in range(len(funcList))]
            loss = sum(lossList) + model.treeNum**(-4)*integration1D(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,[0,1])
            print(_,loss)
            loss.backward()
            del funcList
            del lossList
            del loss
            optimizer.step()


        optimizer = torch.optim.LBFGS(model.treeDict.parameters(), lr=1, max_iter=60)

        def closure():
            optimizer.zero_grad()
            funcList = [lambda x:(sum([integration1D(lambda x:Diffx(Phi(1,n+1,1),x)*Psi(1,j,1)(x), [0,1])*model.treeDict[str(n)](x) - integration1D(lambda x:Phi(1,n+1,1)(x)*Psi(1,j,1)(x),[0,1])*LaplaceOperator(lambda \
                        s:model.treeDict[str(n)](s),x) for n in range(model.treeNum)]) - integration1DforT(
                            lambda s,l:f(s,l)*Psi(order, j, T)(l),[0,T],x)) for j in range(1, model.treeNum+1)]

            #lossList = [Coeff_r(model.treeNum,i+1)*integration1D(lambda x:(funcList[i](x))**2,[0,1]) for i in range(len(funcList))]
            lossList = [integration1D(lambda x:(funcList[i](x))**2,[0,1]) for i in range(len(funcList))]
            loss = sum(lossList) + model.treeNum**(-4)*integration1D(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,[0,1])
            print(loss)
            loss.backward()
            with torch.no_grad():
                outputFunc = lambda x,t: sum([model.treeDict[str(j)](x)*Phi(order,j+1,T)(t) for j in range(model.treeNum)])
                x = torch.linspace(0,1,1000, device='cuda:0').view(1000,1)
                z = outputFunc(x, 0.5).view(1000,1)
                y = real_func(x,torch.tensor(0.5)).view(1000,1)
                print('relerr: {}'.format(torch.norm(y-z)/torch.norm(y)))
                #plt.plot(x.view(1000).cpu(),z.view(1000).cpu())
                #plt.show()
                #plt.plot(x.view(1000).cpu(),y.view(1000).cpu())
                #plt.show()
            return loss

        optimizer.step(closure)
        treeBuffer.append(copy.deepcopy(model.treeDict))


    return treeBuffer
