import torch
import torch.nn as nn
import BinaryTree
from Operations import unary_functions,binary_functions

class Controller(nn.Module):
    def __init__(self, tree, outNum):
        super(Controller, self).__init__()
        self.softmatTemp    = 5.0
        self.tanhC          = 2.5
        self.tree           = tree
        self.batchSize      = 8
        self.NN             = nn.Sequential(
                                nn.Linear(20, 60),
                                nn.ReLU(),
                                nn.Linear(60, 60),
                                nn.ReLU(),
                                nn.Linear(60,self.TotalOP()))
        self.outNum         = outNum

    def probCalc(self):
        inputData = torch.zeros(self.batchSize, 20, requires_grad=True, device= 'cuda:0')
        logits = self.NN(inputData)
        logits = logits/self.softmatTemp
        logits = self.tanhC*torch.tanh(logits)

        return logits

    def sample(self):
        logits = self.probCalc()
        selectedProbLogits = torch.zeros((self.batchSize,0), device='cuda:0')
        inxBuffer = 0
        actions = torch.zeros((self.batchSize,0), dtype=torch.int, device='cuda:0')
        for inx in range(1, self.tree.tree.count+1):
            if BinaryTree.ShowTree(self.tree.tree, inx).is_unary:
                logit = logits[:, inxBuffer:inxBuffer+len(unary_functions)]
                inxBuffer += len(unary_functions)
            else:
                logit = logits[:, inxBuffer:inxBuffer+len(binary_functions)]
                inxBuffer += len(binary_functions)
            prob = nn.functional.softmax(logit, dim=-1)
            action = prob.multinomial(1)

            actions = torch.cat([actions,action],dim=1)
            selectedProbLogits = torch.cat([selectedProbLogits, prob.gather(1, action)], dim=1)
        return actions, selectedProbLogits
    def TotalOP(self):
        t = sum(BinaryTree.TotalopNumCompute(self.tree.tree))
        return t

