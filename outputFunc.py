from Coeff import *
from Equation import *

def intFunc(tree, x, j, T, order):
    out = tree(x) # [-1, model.outNum]
    outNum = out.shape[1]
    laplaceOut = LaplaceOperator(lambda s:tree(s), x)
    intData = 0
    for n in range(1, outNum+1):
        intData = intData + Coeff(j,n,T,'a',order)*out[:,n-1:n] - Coeff(j,n,T,'b',order)*laplaceOut[:,n-1:n]
    return intData

def outputFunc(model,x,t,order,T):
    out = model.tree(x)
    outNum = out.shape[1]
    res = 0
    for j in range(outNum):
        res = res + out[:, j:j+1]*(Phi(order,j+1,T)(t).view(-1,1))
    res = res.view(-1,1)

    return res
