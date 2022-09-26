import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from torchquad import MonteCarlo, set_up_backend, Trapezoid
import BinaryTree
from Equation import LaplaceOperator,RHS4Heat
from Coeff import Coeff
from Controller import Controller
from TreeTrain import TreeTrain
from OperationBuffer import Buffer
from Candidate import Candidate
from Coeff import *
from integration1D import integration1D,integration1DforT
set_up_backend("torch", data_type="float64")
mc = MonteCarlo()
tp = Trapezoid()

def train(model, dim, max_iter, f, real_func):
    order = 1
    T = 1
    domain = [[0,1] for i in range(dim)]
    optimizer4model = torch.optim.Adam(model.NN.parameters())
    buffer = Buffer(3)

    for step in range(max_iter):
        print('-----------step:{}---------------'.format(step))

        actions = model.sample()
        treeBuffer = TreeTrain(f, model, actions, domain, T, dim, 1, real_func)

        errList = torch.zeros(model.batchSize)
        for batch in range(model.batchSize):
            treeDictCompute = treeBuffer[batch]
            funcList = [lambda x:(sum([Coeff(j,n+1,T,'a',1)*treeDictCompute[str(n)](x[:,:]) - Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda \
                        s:treeDictCompute[str(n)](s[:,:]),x) for n in range(model.treeNum)]) - integration1DforT(
                            lambda s,l:f(s,l)*Psi(order, j, T)(l),[0,T],x)) for j in range(1, model.treeNum+1)]

            lossList = [Coeff_r(model.treeNum,i+1)*integration1D(lambda x:(funcList[i](x[:,:]))**2,[0,1]) for i in range(len(funcList))]
            loss = sum(lossList) + model.treeNum**(-4)*integration1D(lambda x:(LaplaceOperator(lambda s:treeDictCompute[str(model.treeNum-1)](s),x))**2,[0,1])
            errList[batch] = loss

        errinx = torch.argmin(errList)
        err = torch.min(errList)
        buffer.refresh(Candidate(treeBuffer[errinx], [actions[i][errinx].cpu().detach().numpy().tolist() for i in range(len(actions))], err.item()))
        optimizer4model.zero_grad()
        #err.backward()
        #optimizer4model.step()
        with torch.no_grad():
            for i in range(len(buffer.bufferzone)):
                print(buffer.bufferzone[i].action, buffer.bufferzone[i].error)
                outputFunc = lambda x,t: sum([buffer.bufferzone[i].treeDict[str(j)](x)*Phi(order,j+1,T)(t) for j in range(model.treeNum)])
                x = torch.linspace(0,1,1000, device='cuda:0').view(1000,1)
                z = outputFunc(x, 0.05).view(1000,1)
                y = real_func(x,torch.tensor(0.5)).view(1000,1)
                print('relerr: {}'.format(torch.norm(y-z)/torch.norm(y)))
                z = outputFunc(x, 0.5).view(1000,1)
                y = real_func(x,torch.tensor(0.5)).view(1000,1)
                print('relerr: {}'.format(torch.norm(y-z)/torch.norm(y)))
                z = outputFunc(x, 0.95).view(1000,1)
                y = real_func(x,torch.tensor(0.5)).view(1000,1)
                print('relerr: {}'.format(torch.norm(y-z)/torch.norm(y)))


if __name__ == '__main__':
    dim = 1
    func = lambda x,t:torch.sin(t)*(torch.exp(x*(x-1))-1)
    f = lambda x,t : RHS4Heat(func,x,t)
    tree = {str(i):BinaryTree.TrainableTree(dim).cuda() for i in range(1)}
    model = Controller(tree).cuda()
    train(model, dim, 50, f, func)
