from Coeff import Phi
def outputFunc(model,x,t,order,T):
    res = 0
    for j in range(len(model.treeDict)):
        res = res + model.treeDict[str(j)](x)*(Phi(order,j+1,T)(t).view(-1,1))

    return res
