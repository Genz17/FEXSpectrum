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
from funcWithVecT import funcTrans
import copy
from outputFunc import outputFunc
set_up_backend("torch", data_type="float64")
mc = MonteCarlo()
tp = Trapezoid()

def train(model, dim, max_iter, f, real_func):
    order = 1
    T = 1
    base = 1e-2
    domain = [[-1,1] for i in range(dim)]
    domainT = copy.deepcopy(domain)
    domainT.append([0,T])
    optimizer4model = torch.optim.Adam(model.NN.parameters())
    buffer = Buffer(5)
    X = 2*(torch.rand((1000,dim), device='cuda:0')-0.5)
    tTest = torch.linspace(0,1,1000, device='cuda:0')

    for step in range(max_iter):
        print('-----------step:{}---------------'.format(step))

        actions, selectedProbLogits = model.sample()
        treeBuffer = TreeTrain(f, model, actions, domain, T, dim, 1, real_func)

        errList = torch.zeros(model.batchSize)
        for batch in range(model.batchSize):
            treeDictCompute = treeBuffer[batch]
            loss = 0
            for j in range(1, model.treeNum+1):
                func = lambda x:sum([Coeff(j,n,T,'a',1)*treeDictCompute[str(n-1)](x) - Coeff(j,n,T,'b',1)*LaplaceOperator(lambda \
                            s:treeDictCompute[str(n-1)](s),x) for n in range(1, model.treeNum+1)])
                tempLoss = mc.integrate(lambda x:((func(x))**2),dim,1000,domain) - \
                        mc.integrate(lambda xt:(func(xt[:,:-1])*(f(xt[:,:-1],xt[:,-1])*(Psi(order,j,T)(xt[:,-1])).view(-1,1))),dim+1,1000,domainT)
                loss = loss + tempLoss
            #loss = loss + 0.1*model.treeNum**(-4)*mc.integrate(lambda x:(LaplaceOperator(lambda s:treeDictCompute[str(model.treeNum-1)](s),x))**2,dim,1000,domain)
            errList[batch] = loss
        errinx = torch.argmin(errList)
        err = torch.min(errList)
        buffer.refresh(Candidate(treeBuffer[errinx], [actions[i][errinx].cpu().detach().numpy().tolist() for i in range(len(actions))], err.item()))

        # Now do the Controller updating...
        #rewards = 1/(1+torch.sqrt(errList))
        #argSortList = torch.argsort(rewards, descending=True)
        #rewardsSorted = rewards[argSortList]
        #lossController = torch.sum(torch.sum(-selectedProbLogits[argSortList][:int(model.batchSize*0.5)],1)*rewardsSorted[:int(model.batchSize*0.5)])
        #print('----------------------------------')
        #print('lossController: {}'.format(lossController))

        #optimizer4model.zero_grad()
        #lossController.backward()
        #optimizer4model.step()
        with torch.no_grad():
            for i in range(len(buffer.bufferzone)):
                print(buffer.bufferzone[i].action, buffer.bufferzone[i].error)
                #outputFunc = lambda x,t: sum([buffer.bufferzone[i].treeDict[str(j)](x)*Phi(order,j+1,T)(t) for j in range(model.treeNum)])
                a = 0
                b = 0
                for tt in range(1000):
                    z = outputFunc(buffer.bufferzone[i],X,tTest[tt],order,T).view(1000,1)
                    y = real_func(X, tTest[tt]).view(1000,1)
                    a = a+torch.norm(z-y, 2)**2
                    b = b+torch.norm(y, 2)**2
                print('relerr: {}'.format(torch.sqrt(a/b)))
        print('---------------------------------')




if __name__ == '__main__':
    dim = 2
    func = lambda xt:torch.exp(torch.sin(2*math.pi*xt[:,-1].view(-1,1))*((torch.prod(xt[:,:-1]**2-1,1)).view(-1,1)))-1
    #func = lambda xt:(xt[:,-1].view(-1,1))*(((xt[:,0]**2-1)*(xt[:,1]**2-1)).view(-1,1))
    f = lambda xt : RHS4Heat(func,xt)
    tree = {str(i):BinaryTree.TrainableTree(dim).cuda() for i in range(8)}
    model = Controller(tree).cuda()
    train(model, dim, 50, f, func)
