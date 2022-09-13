import torch
import copy
from torchquad import MonteCarlo, set_up_backend, Trapezoid
from Equation import LHS_pde,RHS_pde,true_solution,LaplaceOperator
from Coeff import Coeff,Psi,Coeff_r,Coeff_All
set_up_backend("torch", data_type="float32")
mc = MonteCarlo()
tp = Trapezoid()
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
            data = torch.rand(1000,dim)
            funcList = [lambda x:sum([Coeff(j,n+1,T,'a',1)*model.treeDict[str(n)](x)+ Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda\
                        s:model.treeDict[str(n)](s),x) for n in range(model.treeNum)]) - mc.integrate(lambda \
                        t:f(x,t)*Psi(order, j, T)(t),1,integration_domain=[[0,T]]) for j in range(1, model.treeNum+1)]

            loss = sum([Coeff_r(model.treeNum,i+1)*mc.integrate(lambda x:funcList[i](x)**2,dim,1000,domain) for i in range(len(funcList))])+\
                        model.treeNum**(-4)*mc.integrate(lambda x:LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x)**2,dim,1000,domain)
            print(_,loss)
            loss.backward()
            del funcList
            del loss
            optimizer.step()


        optimizer = torch.optim.LBFGS(model.treeDict.parameters(), lr=0.1, max_iter=20)

        def closure():
            optimizer.zero_grad()
            funcList = [lambda x:sum([Coeff(j,n+1,T,'a',1)*model.treeDict[str(n)](x)+Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda\
                        s:model.treeDict[str(n)](s),x) for n in range(model.treeNum)]) - mc.integrate(lambda \
                        t:f(x,t)*Psi(order, j, T)(t),1,1000,integration_domain=[[0,T]]) for j in range(1, model.treeNum+1)]

            loss = sum([Coeff_r(model.treeNum,i+1)*mc.integrate(lambda x:funcList[i](x)**2,dim,1000,domain) for i in range(len(funcList))])+\
                        model.treeNum**(-4)*mc.integrate(lambda x:LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x)**2,dim,1000,domain)
            print(loss)
            loss.backward()
            return loss

        optimizer.step(closure)
        treeBuffer.append(copy.deepcopy(model.treeDict))


    return treeBuffer
