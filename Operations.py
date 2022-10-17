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
    def __init__(self, operator, isLeave):
        super(UnaryOperation, self).__init__()
        self.operator = operator
        self.isLeave  = isLeave
        self.li = nn.Linear(1,1)

    def forward(self, inputData):
        res = self.li(self.operator(inputData))
        return res
class BinaryOperation(nn.Module):
    def __init__(self, operator):
        super(BinaryOperation, self).__init__()
        self.operator = operator
        self.li1 = nn.Linear(1,1)
        self.li2 = nn.Linear(1,1)
        self.li3 = nn.Linear(1,1)

    def forward(self, x, y):
        res = self.li3(self.operator(self.li1(x), self.li2(y)))
        return res



unary_functions = [
    lambda x: 0*x+1,
    lambda x: x,
    lambda x: x**2,
    # lambda x: x**3,
    # lambda x: x**4,
    torch.sin,
    torch.cos,
    torch.exp,
    # lambda x: torch.sin(2*x),
    # lambda x: torch.cos(2*x),
    # lambda x: torch.exp(2*x),
    # lambda y: integration1D(lambda x:torch.exp((x**2-1).view(-1,1))-1,y)
                   ]

binary_functions = [
    lambda x,y: x+y,
    lambda x,y: -x-y,
    lambda x,y: 2*x+y,
    lambda x,y: x+2*y,
    lambda x,y: x*y,
    lambda x,y: -x*y,
    lambda x,y: x-y,
    lambda x,y: -x+y
                    ]


unary_functions_str = ['lambda x: 0*x**2',
                   'lambda x: 1+0*x**2',
                   'lambda x: x+0*x**2',
                   'lambda x: x**2',
                   'lambda x: x**3',
                   'lambda x: x**4',
                   'torch.exp',
                   'torch.sin',
                   'torch.cos',]

binary_functions_str = ['lambda x,y: x+y',
                    'lambda x,y: x*y',
                    'lambda x,y: x-y']


