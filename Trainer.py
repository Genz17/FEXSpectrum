import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchquad import MonteCarlo, set_up_backend
import BinaryTree
from Equation import LHS_pde,RHS_pde,true_solution,LaplaceOperator
from Coeff import Coeff
from DataGen import DataGen
from Controller import Controller
from TreeTrain import TreeTrain
from OperationBuffer import Buffer
from Candidate import Candidate
from Coeff import *
from L2Norm import L2Norm
set_up_backend("torch", data_type="float32")
mc = MonteCarlo()

def train(model, dim, max_iter, f):
    order = 1
    T = 1
    domain = [-1,1]
    optimizer = torch.optim.Adam(model.NN.parameters())
    buffer = Buffer(10)

    for step in range(max_iter):
        print('-----------step:{}---------------'.format(step))

        actions = model.sample()
        treeBuffer = TreeTrain(f, model, actions, domain, 1, dim, 1)
        data = torch.rand((100, dim), device='cuda:0', requires_grad=True)*2-1

        errList = torch.zeros(model.batchSize)
        for batch in range(model.batchSize):
            func = lambda x,j: sum([Coeff(j,treeinx,T,'a', order)*treeBuffer[batch][str(treeinx)](x)+Coeff(j,treeinx,T,'b',order)*LaplaceOperator(treeBuffer[batch][str(treeinx)](x),x,dim)\
for treeinx in range(model.treeNum)])
            funcList = [lambda x:func(x,j)-mc.integrate(lambda t:f(x, t)*Psi(order, j, T)(t), dim=1, integration_domain=[[0,T]]) for j in range(1, model.treeNum+1)]
            lossList = [Coeff_r(model.treeNum, j)*L2Norm(dim, domain, funcList[j-1])**2 for j in range(1,model.treeNum+1)]
            loss = sum(lossList) + model.treeNum**(-4)*L2Norm(dim, domain, lambda x:LaplaceOperator(treeBuffer[batch][str(model.treeNum-1)](x),x,dim))

            errList[batch] = loss

        errinx = torch.argmin(errList)
        err = torch.min(errList)
        buffer.refresh(Candidate(treeBuffer[errinx], [actions[i][errinx].cpu().detach().numpy().tolist() for i in range(model.treeNum)], err.item()))
        optimizer.zero_grad()
        err.backward()
        optimizer.step()
        for i in range(len(buffer.bufferzone)):
            print(buffer.bufferzone[i].action, buffer.bufferzone[i].error)

        with torch.no_grad():
            #x = DataGen(100, dim, -1, 1).cuda()
            x = torch.linspace(-1,1,100, device='cuda:0').view(100, 1).repeat(1,dim)
            z = torch.zeros(x.shape, device='cuda:0')
            #z = 0.5*torch.sum(x**2, 1).view(100,1)
            y = 0
            for i in range(model.treeNum):
                y += (buffer.bufferzone[0].treeDict)[str(i)](x)
            print('relerr: {}'.format(torch.norm(y-z)))
            x = x.view(100).cpu().detach().numpy()
            y = y.view(100).cpu().detach().numpy()
            fig = plt.figure()
            plt.plot(x,y)
            plt.show()


if __name__ == '__main__':
    tree = {str(i):BinaryTree.TrainableTree(1).cuda() for i in range(3)}
    model = Controller(tree).cuda()
    train(model, 1, 10, lambda x,t : x+t)
