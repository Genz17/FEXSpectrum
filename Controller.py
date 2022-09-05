import torch
import torch.nn as nn
import BinaryTree
from Operations import unary_functions,binary_functions

class Controller(nn.Module):
    def __init__(self, treeDict):
        super(Controller, self).__init__()
        self.softmatTemp    = 5.0
        self.tanhC          = 2.5
        self.treeDict       = nn.ModuleDict(treeDict)
        self.treeNum        = len(self.treeDict)
        self.batchSize      = 5
        self.totalChoice()
        self.NN             = nn.Sequential(
                                nn.Linear(20, 60),
                                nn.ReLU(),
                                nn.Linear(60, sum(self.choiceList)))

    def probCalc(self):
        inputData = torch.zeros(self.batchSize, 20, requires_grad=True, device= 'cuda:0')
        logits = self.NN(inputData)
        logits = logits/self.softmatTemp
        logits = self.tanhC*torch.tanh(logits)

        return logits

    def sample(self):
        logits = self.probCalc()
        inxBuffer = 0
        actions = [torch.zeros((self.batchSize,0), dtype=torch.int, device='cuda:0') for i in range(self.treeNum)]
        for i in range(self.treeNum):
            for inx in range(1, self.treeDict[str(i)].tree.count+1):
                if BinaryTree.ShowTree(self.treeDict[str(i)].tree, inx).is_unary:
                    logit = logits[:, inxBuffer:inxBuffer+len(unary_functions)]
                    inxBuffer += len(unary_functions)
                else:
                    logit = logits[:, inxBuffer:inxBuffer+len(binary_functions)]
                    inxBuffer += len(binary_functions)
                prob = nn.functional.softmax(logit, dim=-1)
                action = prob.multinomial(1)

                actions[i] = torch.cat([actions[i],action],dim=1)
        return actions
    def totalChoice(self):
        t = []
        for key in self.treeDict.keys():
            t.append(sum(BinaryTree.TotalopNumCompute(self.treeDict[key].tree)))
        self.choiceList = t
