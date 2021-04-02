# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from FuselageDesign.ShapeFunction import *
from FuselageDesign.ToolKits import *
import FuselageDesign as fd 
#from numpy import logspace,pi,cos 
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt 


# %%

x = np.linspace(0,0.009,13)
x0 = np.linspace(0.01,0.03,9)
x1 = np.linspace(0.03,0.18,11)
x2 = np.linspace(0.18,0.68,21)
x3 = np.linspace(0.69,0.9,15)
nx=[]
nx=np.append(nx,x)
nx=np.append(nx,x0)
nx=np.append(nx,x1[1:])
nx=np.append(nx,x2[1:])
nx=np.append(nx,x3)
nx
plt.plot(nx,zeros(nx.shape),'o-')


# %%
nx

# %% [markdown]
# ### 根据NACA翼型厚度获得宽度曲线

# %%
plt.figure(figsize=[10,2])
x,y=naca4(nx,20)
plt.plot(x/0.9,y,"+-",label="Width")
plt.grid()
plt.axis("equal")
plt.xlabel('x/L')
plt.ylabel('h/L')
plt.legend()
plt.tight_layout()
bb=y

# %% [markdown]
# ### 根据翼型获得高度曲线

# %%
plt.figure(figsize=[10,2])
x,y=getthickness(nx,fn=".\\FuselageDesign\e585.dat")
plt.plot(x/0.9,y,label="High Curve")
plt.axis("equal")
plt.grid()
plt.xlabel('x/L')
plt.ylabel('h/L')
plt.legend()
plt.tight_layout()
aa=y


# %%
## 根据B样条曲线确定nn参数  两组参数


# %%
x=np.linspace(0,1,11)
y1=np.array([2,2.2,3.,4.2,4.6,4.6,4.2,3,2.5,2.2,2])
y2=np.array([2,2.2,2.8,3.4,3.7,3.7,3.5,3,2.5,2.2,2])
from scipy.interpolate import splev, splrep
s1=splrep(x,y1,k=2)
ny1=splev(nx,s1)
plt.plot(x,y1,"o",nx,ny1,label="Upper Shape")
s2=splrep(x,y2,k=2)
ny2=splev(nx,s2)
plt.plot(x,y2,"o",nx,ny2,label="Lower Shape")
plt.legend()
plt.xlabel("x/L")
plt.ylabel("Control Parameters")
nn4=np.array([ny2/2+1,ny1,ny1,ny2])
nn4=nn4.T


# %%
#自制Y曲线

x1=np.array([0, 0.05,  0.18, 0.28, 0.6,1])
y1=np.array([0, 0.01, 0.04, 0.055, 0.07,0.12])

from scipy.interpolate import splev, splrep
s1=splrep(x1,y1,k=2)
ny1=splev(nx,s1)
plt.plot(x1,y1,"o",nx,ny1)
delta = ny1


# %%
#显示测轮廓
of = open("E:\TestFuselageData-2.txt",'w')
psi = np.linspace(0,2*pi,81)
ymax=[]
ymin=[]
zmax=[]
zmin=[]
q=2
nx*=q
for idx, i in enumerate(nx):
    #print(nn4[idx])
    y,z=HyperEllipse4Corner(bb[idx]*1.2,aa[idx],nn4[idx],psi)
    y+=delta[idx]
    y*=q
    z*=q
    ymax.append(y.max())
    ymin.append(y.min())
    zmax.append(z.max())
    zmin.append(z.min())
    plt.figure(figsize=(30,30))  #单位为英寸
    plt.plot(z,y,color='black')
    lables=['x = %d'%i +'\n'+ 'gao   = '  + str(ymax[idx]-ymin[idx])+'\n'+ 'kuan = ' + str(zmax[idx]-zmin[idx])]
    plt.legend(lables,fontsize='xx-large',loc= 'upper left')
    plt.axis("equal")
    plt.savefig('SVG2\%d.svg' % i,format='svg')
    plt.close()
    for jdx, j in enumerate(z):
        of.write("%f %f %f\n"%(i,y[jdx],z[jdx]))
    of.write("\n")
of.close()
plt.figure(figsize=(80,80))
plt.plot(nx,ymax,color='black')
plt.plot(nx,ymin,color='black')
plt.axis("equal")
plt.savefig('SVG2\Side-view2.svg',format='svg')
plt.figure(figsize=(80,80))
plt.plot(nx,zmax,color='black')
plt.plot(nx,zmin,color='black')
plt.axis("equal")
plt.savefig('SVG2\Top-view2.svg',format='svg')


# %%
print(nx)


# %%
len(nx)


