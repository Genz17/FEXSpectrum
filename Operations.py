import torch
import torch.nn as nn

class UnaryOperation(nn.Module):
    def __init__(self, operator, isLeave):
        super(UnaryOperation, self).__init__()
        self.operator = operator
        self.isLeave  = isLeave
        if not isLeave:
            self.li = nn.Linear(1,1)
            nn.init.ones_(self.li.weight)
            nn.init.zeros_(self.li.bias)

    def forward(self, inputData):
        if self.isLeave:
            return self.operator(inputData)
        else:
            return self.li(self.operator(inputData))

class BinaryOperation(nn.Module):
    def __init__(self, operator):
        super(BinaryOperation, self).__init__()
        self.operator = operator

    def forward(self, x, y):
        return self.operator(x, y)




unary_functions = [lambda x: 0*x,
                   lambda x: 1+0*x,
                   lambda x: x,
                   lambda x: x**2,
                   lambda x: x**3,
                   lambda x: x**4,
                   torch.exp,
                   torch.sin,
                   torch.cos,]

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


