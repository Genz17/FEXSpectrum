import torch
def Legendre(n):
    if n == 0:
        return lambda x:0*x+1
    elif n == 1:
        return lambda x:x
    else:
        return lambda x:((2*n-1)/(n))*x*Legendre(n-1)(x) - ((n-1)/n)*Legendre(n-2)(x)


def Phi(order, n, T):
    # starts from 1
    if order == 1:
        return lambda t:Legendre(n)(2*t/T - 1)+Legendre(n-1)(2*t/T - 1)
    if order == 2:
        if n == 1:
            return lambda t:(Legendre(1)(2*t/T - 1)+Legendre(0)(2*t/T - 1))
        else:
            return lambda t:((1-1/n)*Legendre(n)(2*t/T - 1)+(2-1/n)*Legendre(n-1)(2*t/T - 1)+Legendre(n-2)(2*t/T - 1))


def Psi(order, n, T):
    if order == 1:
        if n == 1:
            return lambda t:Legendre(0)(2*t/T - 1)
        else:
            return lambda t:(Legendre(n-1)(2*t/T - 1)-Legendre(n-2)(2*t/T - 1))
    if order == 2:
        if n == 1:
            return lambda t:(Legendre(1)(2*t/T - 1)-Legendre(0)(2*t/T - 1))
        else:
            return lambda t:((1-1/n)*Legendre(n)(2*t/T - 1)-(2-1/n)*Legendre(n-1)(2*t/T - 1)+Legendre(n-2)(2*t/T - 1))


def Coeff_r(treeNum, j):
    return treeNum**(-3)*sum([k*(2*k-1) for k in range(j, treeNum+1)])



def Coeff(j, n, T, coeff_type, order=1):
    if coeff_type == 'a':
        if order == 1:
            if j == 1:
                return 2
            elif j == n + 1:
                return -2
            else:
                return 0
        if order == 2:
            if j == 1:
                return 4*(2*n-1)/(T*n)
            elif j >= 2 and n == 1:
                return -4*(2*n-1)/(T*n)
            elif j == n and n >= 2:
                return 4*(2*n-1)*(n-1)/(T*n)
            else:
                return 0
    if coeff_type == 'b':
        if order == 1:
            if j == n:
                return T/(2*n-1)
            elif j == n+1:
                return -2*T/((2*n-1)*(2*n+1))
            elif j == n+2:
                return -T/(2*n+1)
            else:
                return 0
        if order == 2:
            if j == n and n == 1:
                return -2/3
            elif j == n and n >= 2:
                return -2*T*(2*n-1)*(n**2-n-3)/((n**2)*(2*n-3)*(2*n+1))
            elif j == n+1 and n == 1:
                return 0.5
            elif j == n+1 and n > 1:
                return 2*T/(n*(n+1))
            elif j == n+2 and n == 1:
                return 1/3
            elif j == n+2 and n > 1:
                return T*(n-1)/(n*(2*n+1))
            elif j == n-1:
                return -Coeff(n,j,T,'b',2)
            elif j == n-2:
                return Coeff(n,j,T,'b',2)
            else:
                return 0

def Coeff_All(j, n, T, coeff_type, order):
    res = torch.zeros(n,1, device='cuda:0')
    for i in range(1,n+1):
        res[i-1] = Coeff(j,i,T,coeff_type,order)
    return res

