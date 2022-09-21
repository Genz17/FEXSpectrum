import torch
import numpy as np
import copy
from torchquad import MonteCarlo, set_up_backend, Trapezoid
from Equation import LHS_pde,RHS_pde,true_solution,LaplaceOperator
from Coeff import Coeff,Psi,Coeff_r,Coeff_All
set_up_backend("torch", data_type="float32")
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

        for _ in range(100):
            optimizer.zero_grad()
            if _ % 2 == 1:
                funcList = [lambda x:(sum([Coeff(j,n+1,T,'a',1)*model.treeDict[str(n)](x[:,:]) - Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda \
                            s:model.treeDict[str(n)](s[:,:]),x) for n in range(model.treeNum)]) - mc.integrate(lambda \
                            t:f(x,t)*Psi(order, j, T)(t),1,10000,integration_domain=[[0,T]])) for j in range(1, model.treeNum+1)]

                loss = sum([Coeff_r(model.treeNum,i+1)*mc.integrate(lambda x:(funcList[i](x[:,:]))**2,dim,10000*dim,domain) for i in range(len(funcList))])+\
                            model.treeNum**(-4)*mc.integrate(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,dim,dim*10000,domain)
            else:
                funcList = [lambda x:(sum([Coeff(j,n+1,T,'a',1)*model.treeDict[str(n)](x[:,:]) - Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda \
                            s:model.treeDict[str(n)](s[:,:]),x) for n in range(model.treeNum)]) - mc.integrate(lambda \
                            t:f(x,t)*Psi(order, j, T)(t),1,10000,integration_domain=[[0,T]])) for j in range(1, model.treeNum+1)]

                loss = sum([Coeff_r(model.treeNum,i+1)*mc.integrate(lambda x:(funcList[i](x[:,:]))**2,dim,10000*dim,domain) for i in range(len(funcList))])+\
                            model.treeNum**(-4)*mc.integrate(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,dim,dim*10000,domain)
            print(_,loss)
            loss.backward()
            del funcList
            del loss
            optimizer.step()


        optimizer = torch.optim.LBFGS(model.treeDict.parameters(), lr=0.5, max_iter=40)

        def closure():
            optimizer.zero_grad()
            if torch.rand(1).item() < 0.5:
                funcList = [lambda x:(sum([Coeff(j,n+1,T,'a',1)*model.treeDict[str(n)](x) - Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda\
                            s:model.treeDict[str(n)](s),x) for n in range(model.treeNum)]) - mc.integrate(lambda \
                            t:f(x,t)*Psi(order, j, T)(t),1,10000,integration_domain=[[0,T]])) for j in range(1, model.treeNum+1)]

                loss = sum([Coeff_r(model.treeNum,i+1)*mc.integrate(lambda x:(funcList[i](x))**2,dim,10000*dim,domain) for i in range(len(funcList))])+\
                            model.treeNum**(-4)*mc.integrate(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,dim,dim*10000,domain)
            else:
                funcList = [lambda x:(sum([Coeff(j,n+1,T,'a',1)*model.treeDict[str(n)](x) - Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda\
                            s:model.treeDict[str(n)](s),x) for n in range(model.treeNum)]) - tp.integrate(lambda \
                            t:f(x,t)*Psi(order, j, T)(t),1,10000,integration_domain=[[0,T]])) for j in range(1, model.treeNum+1)]

                loss = sum([Coeff_r(model.treeNum,i+1)*tp.integrate(lambda x:(funcList[i](x))**2,dim,10000*dim,domain) for i in range(len(funcList))])+\
                            model.treeNum**(-4)*tp.integrate(lambda x:(LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x))**2,dim,dim*10000,domain)
            print(loss, loss==np.nan)
            if loss != np.nan:
                loss.backward()
            return loss

        #optimizer.step(closure)
        treeBuffer.append(copy.deepcopy(model.treeDict))


    return treeBuffer
