import torch
import torch.nn as nn

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



unary_functions = [lambda x: 0*x+1,
                   lambda x: 0*x,
                   lambda x: x,
                   lambda x: x**2,
                   lambda x: x**3,
                   lambda x: x**4,
                   torch.sin,
                   torch.cos,
                   torch.exp]

binary_functions = [lambda x,y: x+y,
                    lambda x,y: x*y,
                    lambda x,y: x-y]


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


