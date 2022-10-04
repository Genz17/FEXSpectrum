import torch
import matplotlib.pyplot as plt

def Diffx(func, x, h=5e-4):
    x = x.to(torch.float64)
    u1 = func(x-h)
    u2 = func(x+h)

    return (u2-u1)/(2*h)

def Partialt(func, x, t, h=5e-4):
    x = x.to(torch.float64)
    t = t.to(torch.float64)
    u1 = func(x,t-h)
    u2 = func(x,t+h)

    return (u2-u1)/(2*h)

def LaplaceOperator(func, x, h=5e-4):
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


def LaplaceOperatorWitht(func, x, t, h=5e-4):
    x = x.to(torch.float64)
    t = t.to(torch.float64)
    xNum = x.shape[1]
    s = 0
    u1 = func(x, t)
    for i in range(xNum):
        deltax = torch.zeros_like(x, device='cuda:0')
        deltax[:, i:i+1] = 1
        u2 = func(x+h*deltax, t)
        u3 = func(x-h*deltax, t)
        s = s + (u2+u3-2*u1)/(h**2)
    return s

def RHS4Heat(func, x, t):
    res = Partialt(func,x,t)-LaplaceOperatorWitht(func,x,t)
    return res

