import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from torchquad import MonteCarlo, set_up_backend
import BinaryTree
from Equation import LHS_pde,RHS_pde,true_solution,LaplaceOperator,RHS4Heat
from Coeff import Coeff
from Controller import Controller
from TreeTrain import TreeTrain
from OperationBuffer import Buffer
from Candidate import Candidate
from Coeff import *
set_up_backend("torch", data_type="float32")
mc = MonteCarlo()

def train(model, dim, max_iter, f):
    order = 1
    T = 1
    domain = [[0,1] for i in range(dim)]
    optimizer4model = torch.optim.Adam(model.NN.parameters())
    buffer = Buffer(3)

    for step in range(max_iter):
        print('-----------step:{}---------------'.format(step))

        actions = model.sample()
        treeBuffer = TreeTrain(f, model, actions, domain, T, dim, 1)

        errList = torch.zeros(model.batchSize)
        for batch in range(model.batchSize):
            treeDictCompute = treeBuffer[batch]
            funcList = [lambda x:sum([Coeff(j,n+1,T,'a',1)*treeDictCompute[str(n)](x)-Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda\
                        s:treeDictCompute[str(n)](s),x) for n in range(model.treeNum)]) - mc.integrate(lambda \
                        t:f(x,t)*Psi(order, j, T)(t),1,integration_domain=[[0,T]]) for j in range(1, model.treeNum+1)]

            loss = sum([Coeff_r(model.treeNum,i+1)*mc.integrate(lambda x:funcList[i](x)**2,dim,1000,domain) for i in range(len(funcList))])+\
                        model.treeNum**(-4)*mc.integrate(lambda x:LaplaceOperator(lambda s:model.treeDict[str(model.treeNum-1)](s),x)**2,dim,1000,domain)
            errList[batch] = loss

        errinx = torch.argmin(errList)
        err = torch.min(errList)
        buffer.refresh(Candidate(treeBuffer[errinx], [actions[i][errinx].cpu().detach().numpy().tolist() for i in range(len(actions))], err.item()))
        optimizer4model.zero_grad()
        err.backward()
        optimizer4model.step()
        for i in range(len(buffer.bufferzone)):
            print(buffer.bufferzone[i].action, buffer.bufferzone[i].error)
        if buffer.bufferzone[0].error < 1e-1:
            with torch.no_grad():
                outputFunc = lambda x,t: sum([buffer.bufferzone[0].treeDict[str(j)](x)*Phi(order,j+1,T)(t) for j in range(model.treeNum)])
                x = torch.linspace(0,1,100, device='cuda:0').view(100,1)
                z = outputFunc(x, 1.)
                y = func(x,torch.tensor(1.))
                print('relerr: {}'.format(torch.norm(y-z)/torch.norm(y)))
                plt.plot(x.view(100).cpu(),z.view(100).cpu())
                plt.show()
                plt.plot(x.view(100).cpu(),y.view(100).cpu())
                plt.show()



if __name__ == '__main__':
    dim = 1
    #func = lambda x,t:torch.prod(x,1).view(-1,1)*torch.prod((x-torch.ones_like(x)),1).view(-1,1)
    func = lambda x,t:torch.exp(2*math.pi*t*(((x**2-1)*x**2).view(-1,1)))-1
    tree = {str(i):BinaryTree.TrainableTree(dim).cuda() for i in range(5)}
    model = Controller(tree).cuda()
    train(model, dim, 100, lambda x,t : RHS4Heat(func,x,t))
