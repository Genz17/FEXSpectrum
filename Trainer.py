import torch
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
    optimizer = torch.optim.Adam(model.NN.parameters())
    buffer = Buffer(10)

    for step in range(max_iter):
        print('-----------step:{}---------------'.format(step))

        actions = model.sample()
        treeBuffer = TreeTrain(f, model, actions, domain, 1, dim, 1)

        errList = torch.zeros(model.batchSize)
        for batch in range(model.batchSize):
            funcList = [lambda x:sum([Coeff(j,n+1,T,'a',1)*model.tree(x)[:,n].view(1000,1)+Coeff(j,n+1,T,'b',1)*LaplaceOperator(lambda\
                        s:model.tree(s)[:, n].view(1000,1),x,dim) for n in range(model.tree.outputSize)])- mc.integrate(lambda \
                        t:f(x,t),1,integration_domain=[[0,T]])  for j in range(1, model.tree.outputSize+1)]

            loss = sum([mc.integrate(funcList[i],dim,1000,domain)**2 for i in range(model.tree.outputSize)])
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
    func = lambda x,t:torch.sum(torch.exp(x),1).view(x.shape[0],10)*torch.exp(t)
    tree = BinaryTree.TrainableTree(10, 1).cuda()
    model = Controller(tree).cuda()
    train(model, 10, 100, lambda x,t : RHS4Heat(func,x,t,2))
