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

        errList = torch.zeros(model.batchSize)
        for batch in range(model.batchSize):
            funcList = [lambda x:sum([Coeff(j,n+1,T,'a',1)*treeBuffer[batch](x)[:,n].view(100,1)+Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda s:treeBuffer[batch](s)[:, n].view(100,1),x,dim) for n in range(model.tree.outputSize)])- mc.integrate(lambda \
                        t:f(x,t),1,integration_domain=[[0,T]])  for j in range(1, model.tree.outputSize+1)]

            loss = sum([mc.integrate(funcList[i],dim,100,[domain])**2 for i in range(model.tree.outputSize)])
            errList[batch] = loss

        errinx = torch.argmin(errList)
        err = torch.min(errList)
        buffer.refresh(Candidate(treeBuffer[errinx], actions[errinx].cpu().detach().numpy().tolist(), err.item()))
        optimizer.zero_grad()
        err.backward()
        optimizer.step()
        for i in range(len(buffer.bufferzone)):
            print(buffer.bufferzone[i].action, buffer.bufferzone[i].error)

        #with torch.no_grad():
        #    #x = DataGen(100, dim, -1, 1).cuda()
        #    x = torch.linspace(-1,1,100, device='cuda:0').view(100, 1).repeat(1,dim)
        #    z = torch.zeros(x.shape, device='cuda:0')
        #    #z = 0.5*torch.sum(x**2, 1).view(100,1)
        #    y = (buffer.bufferzone[0].tree)(x)
        #    print('relerr: {}'.format(torch.norm(y-z)))
        #    x = x.view(100).cpu().detach().numpy()
        #    y = y.view(100).cpu().detach().numpy()
        #    fig = plt.figure()
        #    plt.plot(x,y)
        #    plt.show()


if __name__ == '__main__':
    tree = BinaryTree.TrainableTree(1, 1).cuda()
    model = Controller(tree).cuda()
    train(model, 1, 10, lambda x,t : torch.norm(x)+t)
