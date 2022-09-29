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
from funcCoeffList import funcCoeffListGen
from Coeff import *
from integration1D import integration1DforT
set_up_backend("torch", data_type="float64")
mc = MonteCarlo()
tp = Trapezoid()

def train(model, dim, max_iter, f, real_func):
    order = 1
    T = 1
    domain = [[-1,1] for i in range(dim)]
    optimizer4model = torch.optim.Adam(model.NN.parameters())
    buffer = Buffer(5)
    X = 2*(torch.rand((1000,dim), device='cuda:0')-0.5)
    tTest = torch.linspace(0,1,1000, device='cuda:0')

    for step in range(max_iter):
        print('-----------step:{}---------------'.format(step))

        actions = model.sample()
        treeBuffer = TreeTrain(f, model, actions, domain, T, dim, 1, real_func)

        errList = torch.zeros(model.batchSize)
        for batch in range(model.batchSize):
            treeDictCompute = treeBuffer[batch]
            loss = 0
            for j in range(1, model.treeNum+3):
                func = lambda x:(sum([Coeff(j,n,T,'a',1)*treeDictCompute[str(n-1)](x) - Coeff(j,n,T,'b',1)*LaplaceOperator(lambda \
                            s:treeDictCompute[str(n-1)](s),x) for n in funcCoeffListGen(j, model.treeNum,1)]) - integration1DforT(
                                lambda s,l:f(s,l)*Psi(order, j, T)(l),T,x))
                loss = loss + Coeff_r(model.treeNum,j)*tp.integrate(lambda x:(func(x))**2,dim,1000,domain)
            loss = loss + 0.1*model.treeNum**(-4)*tp.integrate(lambda x:(LaplaceOperator(lambda s:treeDictCompute[str(model.treeNum-1)](s),x))**2,dim,1000,domain)
            errList[batch] = loss

        errinx = torch.argmin(errList)
        err = torch.min(errList)
        buffer.refresh(Candidate(treeBuffer[errinx], [actions[i][errinx].cpu().detach().numpy().tolist() for i in range(len(actions))], err.item()))
        optimizer4model.zero_grad()
        err.backward()
        optimizer4model.step()
        with torch.no_grad():
            for i in range(len(buffer.bufferzone)):
                print(buffer.bufferzone[i].action, buffer.bufferzone[i].error)
                outputFunc = lambda x,t: sum([buffer.bufferzone[i].treeDict[str(j)](x)*Phi(order,j+1,T)(t) for j in range(model.treeNum)])
                a = 0
                b = 0
                for tt in range(1000):
                    z = outputFunc(X, tTest[tt]).view(1000,1)
                    y = real_func(X, tTest[tt]).view(1000,1)
                    a = a+torch.norm(z-y, 2)**2
                    b = b+torch.norm(y, 2)**2
                print('relerr: {}'.format(torch.sqrt(a/b)))



if __name__ == '__main__':
    dim = 2
    func = lambda x,t:torch.exp(torch.sin(2*math.pi*t)*(((x[:,0]**2-1)*(x[:,1]**2-1)).view(-1,1)))-1
    #func = lambda x,t:torch.sin(2*math.pi*t)*(((x+1)*(x-1)).view(-1,1))
    f = lambda x,t : RHS4Heat(func,x,t)
    tree = {str(i):BinaryTree.TrainableTree(dim).cuda() for i in range(8)}
    model = Controller(tree).cuda()
    train(model, dim, 50, f, func)
