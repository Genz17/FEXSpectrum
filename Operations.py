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
    def __init__(self, operator, inNum, outNum):
        super(UnaryOperation, self).__init__()
        self.operator = operator
        self.li = nn.Linear(inNum,outNum)

    def forward(self, inputData):
        res = self.li(self.operator(inputData))
        return res
class BinaryOperation(nn.Module):
    def __init__(self, operator, inNum, outNum):
        super(BinaryOperation, self).__init__()
        self.operator = operator
        self.li1 = nn.Linear(inNum,outNum)

    def forward(self, x, y):
        res = self.li1(self.operator(x, y))
        return res



unary_functions = [
    lambda x: x,
    lambda x: x**2,
    lambda x: x**3,
    lambda x: x**4,
    torch.sin,
    torch.cos,
    torch.exp
    # lambda y: integration1D(lambda x:torch.exp((x**2-1).view(-1,1))-1,y)
                   ]

binary_functions = [
    lambda x,y: x+y,
    lambda x,y: x*y,
    lambda x,y: x-y,
    lambda x,y: torch.exp(x+y),
    lambda x,y: torch.sin(x+y),
    lambda x,y: torch.cos(x+y)
    # lambda x,y: torch.relu_(x) + torch.relu(y)
    # lambda x,y: x**2+y**2,
    # lambda x,y: x**3+y**3,
    # lambda x,y: x**4+y**4,
    # lambda x,y: torch.exp(x+y),
    # lambda x,y: torch.sin(x+y),
    # lambda x,y: torch.cos(x+y)
                    ]


unary_functions_str = [
                    'lambda x: x',
                    'lambda x: x**2',
                    'lambda x: x**3',
                    'lambda x: x**4',
                    'torch.sin',
                    'torch.cos',
                    'torch.exp'
                    ]

binary_functions_str = ['lambda x,y: x+y',
                    'lambda x,y: x*y',
                    'lambda x,y: x-y',
                    'lambda x,y: torch.exp(x+y)',
                    'lambda x,y: torch.sin(x+y)',
                    'lambda x,y: torch.cos(x+y)'
                    ]


