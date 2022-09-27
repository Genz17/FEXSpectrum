def funcCoeffListGen(j, nNum, order):
    #print('j = {}'.format(j))
    if order == 1:
        if j == 1:
            return range(1, nNum+1)
        if j == 2:
            return list(set([1,2]) & set(range(1,nNum+1)))
        else:
            return list(set([j-2, j-1, j]) & set(range(1,nNum+1)))
