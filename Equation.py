import torch
import matplotlib.pyplot as plt

def Diffx(func, x, h=2e-4):
    x = x.to(torch.float64)
    u1 = func(x-h)
    u2 = func(x+h)

    return (u2-u1)/(2*h)

def Partialt(func, x, t, h=2e-4):
    t = t.to(torch.float64)
    x = x.to(torch.float64)
    u1 = func(x,t-h)
    u2 = func(x,t+h)

    return (u2-u1)/(2*h)

def LaplaceOperator(func, x, h=2e-4):
    x = x.to(torch.float64)
    xNum = x.shape[1]
    s = torch.zeros_like(x)
    u1 = func(x)
    for i in range(xNum):
        deltax = torch.zeros_like(x, device='cuda:0')
        deltax[:, i:i+1] = 1
        u2 = func(x+h*deltax)
        u3 = func(x-h*deltax)
        s = s + (u2+u3-2*u1)/(h**2)
    return s


def LaplaceOperatorWitht(func, x, t, h=2e-4):
    t = t.to(torch.float64)
    x = x.to(torch.float64)
    xNum = x.shape[1]
    s = torch.zeros_like(x)
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

#def RHS4Heat(func, x, t, dim):
#    x.requires_grad = True
#    t.requires_grad = True
#    u = func(x, t)
#    v = torch.ones(u.shape, device='cuda:0')
#    ut = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
#    bs = x.size(0)
#    ux = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True)[0]
#    uxx = torch.zeros(bs, dim, device='cuda:0')
#    for i in range(dim):
#        ux_tem = ux[:, i:i+1]
#        uxx_tem = torch.autograd.grad(ux_tem, x, grad_outputs=torch.ones_like(ux_tem), create_graph=True)[0]
#        uxx[:, i] = uxx_tem[:, i]
#    LHS = -torch.sum(uxx, dim=1, keepdim=True)
#    return ut+LHS
#
#def LaplaceOperator(func, x, dim):
#    x.requires_grad = True
#    u = func(x)
#    v = torch.ones(u.shape, device='cuda:0')
#    bs = x.size(0)
#    ux = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True)[0]
#    uxx = torch.zeros(bs, dim, device='cuda:0')
#    for i in range(dim):
#        ux_tem = ux[:, i:i+1]
#        uxx_tem = torch.autograd.grad(ux_tem, x, grad_outputs=v, create_graph=True)[0]
#        uxx[:, i] = uxx_tem[:, i]
#    LHS = -torch.sum(uxx, dim=1, keepdim=True)
#    return LHS


def LHS_pde(u, x, dim_set):

    v = torch.ones(u.shape, device='cuda:0')
    bs = x.size(0)
    ux = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True)[0]
    uxx = torch.zeros(bs, dim_set, device='cuda:0')
    for i in range(dim_set):
        ux_tem = ux[:, i:i+1]
        uxx_tem = torch.autograd.grad(ux_tem, x, grad_outputs=v, create_graph=True)[0]
        uxx[:, i] = uxx_tem[:, i]
    LHS = -torch.sum(uxx, dim=1, keepdim=True)
    return LHS

def RHS_pde(x):
    bs = x.size(0)
    dim = x.size(1)
    return -dim*torch.ones(bs, 1).cuda()

def true_solution(x):
    return 0.5*torch.sum(x**2, dim=1, keepdim=True).cuda()#1 / (2 * x[:, 0:1] + x[:, 1:2]-5)
