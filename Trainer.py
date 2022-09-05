import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import BinaryTree
from Equation import LHS_pde,RHS_pde,true_solution,LaplaceOperator
from Coeff import Coeff
from DataGen import DataGen
from Controller import Controller
from GetReward import GetReward
from OperationBuffer import Buffer
from Candidate import Candidate

def train(model, dim, max_iter):
    optimizer = torch.optim.Adam(model.NN.parameters())
    buffer = Buffer(10)

    for step in range(max_iter):
        print('-----------step:{}---------------'.format(step))

        actions = model.sample()
        treeBuffer = GetReward(model, actions, -1, 1, 1, dim, 1)
        data = torch.rand((100, dim), device='cuda:0', requires_grad=True)*2-1

        errList = torch.zeros(model.batchSize)
        for batch in range(model.batchSize):
            loss = 0
            for j in range(model.treeNum):
                ans = 0
                for treeinx in range(model.treeNum):
                    res = treeBuffer[batch][str(treeinx)](data)
                    ans += Coeff(j, treeinx, 1, 'a', order=1)*res
                    ans += Coeff(j, treeinx, 1, 'b', order=1)*LaplaceOperator(res, data, dim)
                loss += torch.norm(ans)

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
            #x = x.view(100).cpu().detach().numpy()
            #y = y.view(100).cpu().detach().numpy()
            #fig = plt.figure()
            #plt.plot(x,y)
            #plt.show()


if __name__ == '__main__':
    tree = {str(i):BinaryTree.TrainableTree(10).cuda() for i in range(1)}
    model = Controller(tree).cuda()
    train(model, 10, 10)
