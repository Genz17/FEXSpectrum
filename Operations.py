import torch
import torch.nn as nn
import math
from torchquad import MonteCarlo, set_up_backend, Boole
from integration1D import integration1D
torch.set_default_tensor_type('torch.cuda.DoubleTensor')
set_up_backend("torch", data_type="float64")
mc = MonteCarlo()
bl = Boole()

class UnaryOperation(nn.Module):
    def __init__(self, operator, inNum, outNum, treeCount):
        super(UnaryOperation, self).__init__()
        self.treeCount = treeCount
        self.operator = operator
        self.li = nn.Linear(inNum,outNum)

    def forward(self, inputData):
        res = self.li(self.operator(inputData))
        # print('Node :{}: input: {}; output: {}'.format(self.treeCount, torch.norm(inputData), torch.norm(res)))
        return res
class BinaryOperation(nn.Module):
    def __init__(self, operator, inNum, outNum, treeCount):
        super(BinaryOperation, self).__init__()
        self.treeCount = treeCount
        self.operator = operator
        self.li1 = nn.Linear(inNum,outNum)

    def forward(self, x, y):
        res = self.li1(self.operator(x, y))
        # print('Node: {}, inputx: {}; inputy: {}; output: {}'.format(self.treeCount, torch.norm(x), torch.norm(y), torch.norm(res)))
        return res



unary_functions = [
    lambda x: x,
    lambda x: x**2,
    lambda x: x**3,
    lambda x: x**4,
    torch.sin,
    torch.cos,
    torch.exp,
    lambda x:torch.exp(torch.sin(x)),
    lambda x:torch.exp(torch.cos(x)),
                   ]

binary_functions = [
    lambda x,y: x+y,
    lambda x,y: x*y,
    lambda x,y: x-y,
    lambda x,y: torch.sin(x+y),
    lambda x,y: torch.cos(x+y),
    lambda x,y: torch.exp(x+y),
    lambda x,y: torch.exp(torch.sin(x+y)),
    lambda x,y: torch.exp(torch.cos(x+y)),
                    ]


unary_functions_str = [
                    'x',
                    'x**2',
                    'x**3',
                    'x**4',
                    'sin',
                    'cos',
                    'exp',
                    'exp(sin)',
                    'exp(cos)',
                    ]

binary_functions_str = [
                    'x+y',
                    'x*y',
                    'x-y',
                    'sin(x+y)',
                    'cos(x+y)',
                    'exp(x+y)',
                    'exp(sin(x+y))',
                    'exp((x+y))',
                    ]


