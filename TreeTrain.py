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
        batchOperation = batchOperations[batch]
        model.tree.PlaceOP(batchOperation)
        model.tree.LinearGen()

        optimizer = torch.optim.Adam(model.tree.parameters())

        for _ in range(20):
            optimizer.zero_grad()
            funcList = [lambda x:torch.sum(Coeff_All(j,model.tree.outputSize,T,'a',1).view(1,-1).repeat(100,1)*model.tree(x)+Coeff_All(j,model.tree.outputSize,T,'b',1).view(1,-1).repeat(100,1)*LaplaceOperator(lambda\
                        s:model.tree(s),x,dim), dim=1)- mc.integrate(lambda \
                        t:f(x,t),1,100,integration_domain=[[0,T]])  for j in range(1, model.tree.outputSize+1)]
            loss = sum([mc.integrate(funcList[i],dim,100,domain)**2 for i in range(model.tree.outputSize)])
            loss.backward()
            optimizer.step()
            print(_)


        optimizer = torch.optim.LBFGS(model.tree.parameters(), lr=1, max_iter=20)

        def closure():
            optimizer.zero_grad()
            funcList = [lambda x:torch.sum(Coeff_All(j,model.tree.outputSize,T,'a',1).view(1,-1).repeat(1000,1)*model.tree(x)+Coeff_All(j,model.tree.outputSize,T,'b',1).view(1,-1).repeat(1000,1)*LaplaceOperator(lambda\
                        s:model.tree(s),x,dim), dim=1)- mc.integrate(lambda \
                        t:f(x,t),1,integration_domain=[[0,T]])  for j in range(1, model.tree.outputSize+1)]
            loss = sum([mc.integrate(funcList[i],dim,1000,domain)**2 for i in range(model.tree.outputSize)])
            loss.backward()
            return loss

        #optimizer.step(closure)
        treeBuffer.append(copy.deepcopy(model.tree))


    return treeBuffer
