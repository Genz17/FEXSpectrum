import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from torchquad import MonteCarlo, set_up_backend, Boole
import BinaryTree
from Equation import LaplaceOperator,RHS4Heat
from Coeff import Coeff
from Controller import Controller
from TreeTrain import TreeTrain
from OperationBuffer import Buffer
from Candidate import Candidate
from funcCoeffList import funcCoeffListGen
from Coeff import *
from funcWithVecT import funcTrans
import copy
from outputFunc import intFunc,outputFunc
torch.set_default_tensor_type('torch.cuda.DoubleTensor')
set_up_backend("torch", data_type="float64")
mc = MonteCarlo()
bl = Boole()

def train(model, dim, max_iter, f, real_func):
    order = 1
    T = 1.0
    domain = [[-1,1] for i in range(dim)]
    domainT = copy.deepcopy(domain)
    domainT.append([0,T])
    optimizer4model = torch.optim.Adam(model.NN.parameters(), lr=1e-1)
    buffer = Buffer(5)
    X = 2*(torch.rand((1000,dim), device='cuda:0')-0.5)
    tTest = torch.rand((1000,1), device='cuda:0')*T
    XT = torch.zeros((1000,dim+1),device='cuda:0')
    XT[:,:-1] = X
    XT[:,-1] = tTest.view(1000)

    for step in range(max_iter):
        print('-----------step:{}---------------'.format(step))

        actions, selectedProbLogits = model.sample()
        print(selectedProbLogits)
        treeBuffer = TreeTrain(f, model, actions, domain, T, dim, 1, real_func)

        errList = torch.zeros(model.batchSize)
        with torch.no_grad():
            for batch in range(model.batchSize):
                treeDictCompute = treeBuffer[batch]
                loss = 0
                for j in range(1, model.outNum+1):
                    tempLoss = mc.integrate(lambda x:((intFunc(treeDictCompute, x, j, T, order))**2),dim,10000,domain,seed=1) - \
                            mc.integrate(lambda xt:2*(intFunc(treeDictCompute, xt[:,:-1], j, T, order)*(f(xt)*(Psi(order,j,T)(xt[:,-1])).view(-1,1))),dim+1,10000,domainT,seed=1)
                    loss = loss + Coeff_r(model.outNum,j)*tempLoss
                    #loss = loss + 0.1*model.treeNum**(-4)*mc.integrate(lambda x:(LaplaceOperator(lambda s:treeDictCompute[str(model.treeNum-1)](s),x))**2,dim,1000,domain)
                errList[batch] = loss
            errinx = torch.argmin(errList)
            err = torch.min(errList)
        buffer.refresh(Candidate(treeBuffer[errinx], actions[errinx].cpu().detach().numpy().tolist(), err.item()))

        # Now do the Controller updating...
        for i in range(model.batchSize):
            if errList[i].item() >= 0:
                errList[i] = 0
        rewards = nn.functional.softmax(torch.abs(errList))
        print('rewards:{}'.format(rewards))
        argSortList = torch.argsort(rewards, descending=True)
        rewardsSorted = rewards[argSortList]
        lossController = torch.sum(torch.sum(-selectedProbLogits[argSortList][:int(model.batchSize*0.5)],1)*rewardsSorted[:int(model.batchSize*0.5)])
        print('----------------------------------')
        print('lossController: {}'.format(lossController))

        optimizer4model.zero_grad()
        lossController.backward()
        optimizer4model.step()
        with torch.no_grad():
            for i in range(len(buffer.bufferzone)):
                print(buffer.bufferzone[i].action, buffer.bufferzone[i].error)
                z = outputFunc(buffer.bufferzone[i],X,tTest,order,T).view(1000,1)
                y = real_func(XT).view(1000,1)
                a = (z-y)**2
                b = y**2
                a = torch.sum(a,0)
                b = torch.sum(b,0)
                print('relerr: {}'.format(torch.sqrt(a/b).item()))
        print('---------------------------------')


if __name__ == '__main__':
    dim = 2
    outNum = 16
    func = lambda xt:torch.exp(torch.sin(2*math.pi*xt[:,-1].view(-1,1))*((torch.prod(xt[:,:-1]**2-1,1)).view(-1,1)))-1
    print('We are using func exp(sin(2\\pi t)\\Prod(x_i^2-1).')
    # func = lambda xt:torch.sin(2*math.pi*xt[:,-1].view(-1,1))*((torch.prod(xt[:,:-1]**2-1,1)).view(-1,1))
    # func = lambda xt:(xt[:,-1].view(-1,1))*(torch.prod(xt[:,:-1]**2-1,1)).view(-1,1)
    f = lambda xt : RHS4Heat(func,xt)
    # f = lambda xt:(torch.prod(xt[:,:-1]**2-1,1)).view(-1,1) - 2*((xt[:,-1])*((xt[:,0]**2-1)+(xt[:,1]**2-1))).view(-1,1)

    tree = BinaryTree.TrainableTree(dim, outNum).cuda()
    model = Controller(tree, outNum).cuda()
    train(model, dim, 500, f, func)
