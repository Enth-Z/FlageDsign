def guessratio(n,t,f):
    """根据等比数列首数、尾数、数列长度以及数字总和，猜测等比数列比值。
       Input:
            f    : 等比数列第一个数字
            n    : 等比数列数字个数
            t    : 等比数列中所有数字和
       Output 
            f    : 等比数列最后一个数字
            r    : 比值
       """
    if f==0.: f=1e-6
    if t / (f * n) >= 1. :
        A = 1.
        B = (t / n) / f + 0.01
    else:
        A = 0.
        B = 1.
    err_sum = t - f * n
    r = 1.
    counter=0
    while ( counter<100 and abs(err_sum)>1E-8):
        counter += 1
        r = (A + B) / 2.
        err_sum = abs(t) - abs(f)*(1.-r**n)/(1.-r)
        if err_sum > 0.:
            A = r
        else:
            B = r
    l = f * r**(n-1)
    return n,t,f,l,r