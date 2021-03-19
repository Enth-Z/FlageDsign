from numpy import cos,sin,pi,zeros,sqrt,loadtxt,append,zeros_like,arange,where,sqrt
from numpy.core.function_base import linspace
from scipy import interpolate
import numpy as np

def HyperEllipse( aa, bb, nn, psi, half=False):     
    """超椭圆函数 
       aa:高向半长轴、
       bb:宽向半长轴、
       nn：次方数、1 菱形，2正圆，5老电视
       psi：角度值（弧度、0-2pi）,建议数字为4n+1
    """
    na = 2.0/nn
    tt = psi
    sgn = zeros(tt.shape)
    mask = cos(tt)>0.
    sgn[mask] = 1.
    mask = cos(tt)<0.
    sgn[mask] = -1.
    x = (abs((cos(tt)))**na)*aa * sgn

    sgn = zeros(tt.shape)
    mask = sin(tt)>0.
    sgn[mask] = 1.
    mask = sin(tt)<0.
    sgn[mask] = -1.
    y = (abs((sin(tt)))**na)*bb * sgn
    return[y,x]

def HyperEllipse4Corner( aa, bb, nn, psi, half=False):     
    """超椭圆函数 四个角外形分别可控
       aa:高向半长轴、
       bb:宽向半长轴、
       nn：指定超椭圆函数的次方数。可以按照三种形式给，
           形式一： n， 整数n取值范围使外形从内凹的星型->菱形->圆形->圆角方形， 整数值 1=菱形，2=正圆，5=老电视
           形式二： [n1,n2], 右部分外形按照n1计算，左部分按照n2计算
           形式三： [n1,n2,n3,n4], 四个角分别按照n1到n4计算，分别为右上、右下、左下和左上。
       psi：角度值（弧度、0-2pi）,建议数字为4n+1
    """
    #na = 2.0/nn
    na = zeros(psi.shape)
    if len(nn)==1:
        na = 2.0/nn
    elif len(nn)==2:
        q = psi.shape[0]//2
        for idx, i in enumerate(nn):
            na[idx*q:(idx+1)*q] = 2.0/i
    elif len(nn)==4:
        q = psi.shape[0]//2
        for idx, i in enumerate(nn):
            na[idx*q:(idx+1)*q] = 2.0/i
    else:
        na = 2.0/nn 
    tt = psi
    sgn = zeros(tt.shape)
    mask = cos(tt)>0.
    sgn[mask] = 1.
    mask = cos(tt)<0.
    sgn[mask] = -1.
    x = (abs((cos(tt)))**na)*aa * sgn

    sgn = zeros(tt.shape)
    mask = sin(tt)>0.
    sgn[mask] = 1.
    mask = sin(tt)<0.
    sgn[mask] = -1.
    y = (abs((sin(tt)))**na)*bb * sgn
    return[y,x]




def Circle(r,psi):
    """创造圆形 
       r为半径；
       psi为方位角"""
    return [r*sin(psi), r*cos(psi)]

def naca4(x,thickness):    
    """NACA四位数翼型厚度生成函数
       x: 一维数组，必须保证其中数字取值范围0-1。
       thickness： 最大厚度除弦长的百分比，如12。
    """
    x = x
    #tt = parameters['tt']
    tt = thickness*0.01
    
    y = (tt/0.2)*(0.2969*sqrt(x) - 0.126*x - 0.3516*x*x + 0.2843*x*x*x -0.1015*x*x*x*x)
    return [x,y]

def airfoildata(x,fn='e585.dat',lex=0.015):
    """采用B样条方法插值，为保证前缘光滑，在x=0.01 局部加密插值"""
    from scipy.interpolate import splev, splrep
    dataOrg = loadtxt(fn,skiprows=1)
    xo, yo = dataOrg[:,0],dataOrg[:,1] 
    cnt = 0
    print(len(xo))
    for i in range(len(xo)-1):
        if xo[i]>=xo[i+1]:
            cnt = i
    if xo[cnt]!=0.0:
        xou = append(xo[:cnt+1], 0.)
        you = append(yo[:cnt+1], 0.)
        xol = append(0., xo[cnt+1:])
        yol = append(0., yo[cnt+1:])
    if yol.mean() > you.mean():
        xou,you,xol,yol = xol,yol,xou,you
    
    xou,you = xou[::-1],you[::-1]
    nheadu = (xou<lex).sum()

    spluhead = splrep(xou[:nheadu]*10, you[:nheadu]*10)
    splutail = splrep(xou[nheadu-4:], you[nheadu-4:])

    nheadl = (xol<lex).sum()
    spllhead = splrep(xol[:nheadl]*10, yol[:nheadl]*10)
    splltail = splrep(xol[nheadl-4:], yol[nheadl-4:])

    mahead = (x<lex).sum()
    yu = zeros_like(x)
    yl = zeros_like(x)

    yu[:mahead] = splev(x[:mahead]*10.,spluhead)/10.
    yl[:mahead] = splev(x[:mahead]*10.,spllhead)/10.

    yu[mahead:] = splev(x[mahead:],splutail)
    yl[mahead:] = splev(x[mahead:],splltail)

    return [x,yu-yl]

def airfoildata2(fn='e585.dat',n=1000):
    """采用B样条方法插值,x,y"""
    from scipy.interpolate import splev, splrep
    dataOrg = loadtxt(fn,skiprows=1)
    xo, yo = dataOrg[:,0],dataOrg[:,1] 

    splx = splrep(arange(len(xo)), xo)
    sply = splrep(arange(len(yo)), yo)

    x = zeros(n)
    y = zeros(n)

    x = splev(linspace(0,len(xo),n),splx)
    y = splev(linspace(0,len(yo),n),sply)
    return [x,y]

def getthickness(x,fn=".\\FuselageDesign\e585.dat"):
    from scipy import interpolate

    dataOrg = loadtxt(fn,skiprows=1)
    xo, yo = dataOrg[:,0],dataOrg[:,1] 
    ix,iy=airfoildata2(fn,n=1500)
    idx = int(where(ix==ix.min())[0])

    ux,uy = ix[:idx],iy[:idx]
    lx,ly = ix[idx-1:],iy[idx-1:]
    uf = interpolate.interp1d(ux,uy,kind="linear")
    lf = interpolate.interp1d(lx,ly,kind="linear")
    if  ly.mean()>uy.mean():
        uf,lf = lf,uf
    return [x,uf(x)-lf(x)]

def vitosinski (x,r1,r2,L):
    """委托辛斯基曲线 进口半径

    Args:
        x (float array): 横坐标
        r1 (float): 进口半径
        r2 (float): 出口半径
        L (float): 进口出口距离

    Returns:
        [type]: [description]
    """    
    r = r2/sqrt(1-(1-(r2/r1)**2)*(1-x**2/L**2)**2/(1+x**2/3/L**2)**3)
    return [x,r]

def zhixian(x,k1,k2):
    r=[]
    for i in range(0,len(x)):
        r1 = i*k2+k1
        r=np.append(r,r1)
    return [x,r]

def erciquxian(x,k1,k2):
    r=[]
    for i in range(0,len(x)):
        r1=x[i]**k1*k2
        r=np.append(r,r1)
    return [x,r]