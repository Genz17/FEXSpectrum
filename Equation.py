import torch
import matplotlib.pyplot as plt
torch.set_default_tensor_type('torch.cuda.DoubleTensor')

def Diffx(func, x, h=5e-5):
    x = x.to(torch.float64)
    u1 = func(x-h)
    u2 = func(x+h)

    return (u2-u1)/(2*h)

def Partialt(func, xt, h=5e-5):
    xt = xt.to(torch.float64)
    deltat = torch.zeros_like(xt)
    deltat[:,-1] = 1
    u1 = func(xt - h*deltat)
    u2 = func(xt + h*deltat)

    return (u2-u1)/(2*h)

#def Partialt(func, x, t, h=5e-4):
#    x = x.to(torch.float64)
#    t = t.to(torch.float64)
#    u1 = func(x,t-h)
#    u2 = func(x,t+h)
#
#    return (u2-u1)/(2*h)

def LaplaceOperator(func, x, h=5e-5):
    x = x.to(torch.float64)
    xNum = x.shape[1]
    s = 0
    u1 = func(x)
    for i in range(xNum):
        deltax = torch.zeros_like(x, device='cuda:0')
        deltax[:, i:i+1] = 1
        u2 = func(x+h*deltax)
        u3 = func(x-h*deltax)
        s = s + (u2+u3-2*u1)/(h**2)
    return s


def LaplaceOperatorWitht(func, xt, h=5e-5):
    xt = xt.to(torch.float64)
    xtNum = xt.shape[1]-1
    s = 0
    u1 = func(xt)
    for i in range(xtNum):
        deltax = torch.zeros_like(xt, device='cuda:0')
        deltax[:, i:i+1] = 1
        u2 = func(xt+h*deltax)
        u3 = func(xt-h*deltax)
        s = s + (u2+u3-2*u1)/(h**2)
    return s
#def LaplaceOperatorWitht(func, x, t, h=5e-4):
#    x = x.to(torch.float64)
#    t = t.to(torch.float64)
#    xNum = x.shape[1]
#    s = 0
#    u1 = func(x, t)
#    for i in range(xNum):
#        deltax = torch.zeros_like(x, device='cuda:0')
#        deltax[:, i:i+1] = 1
#        u2 = func(x+h*deltax, t)
#        u3 = func(x-h*deltax, t)
#        s = s + (u2+u3-2*u1)/(h**2)
#    return s
#
#def RHS4Heat(func, x, t):
#    res = Partialt(func,x,t)-LaplaceOperatorWitht(func,x,t)
#    return res
def RHS4Heat(func, xt):
    res = Partialt(func,xt)-LaplaceOperatorWitht(func,xt)
    return res

