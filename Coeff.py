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


