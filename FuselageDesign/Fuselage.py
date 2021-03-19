import matplotlib.pyplot as plt 
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np 
import scipy
import csv

from numpy import cos, sin
from pyqtgraph.Qt import QtCore, QtGui

class Fuselage():
    '''  机身类定义,机身坐标参照飞机坐标系设定，飞行员背向X，右手平直指向Z，向上为Y轴。机头x坐标为0.0'''
    def __init__(self,length=2.0,xresolution=60,a=0.08,b=0.08,n=6.0,yd=0,sf=1.4,
                 a0=0,a1=0,a2=0,a3=0,a4=0,a5=0):
        '''length 机身长度， xresolution 纵向剖面数量， '''
        #机身长度
        self.length = float(length)
        #x方向设计分辨率，aka 设计轮廓线数
        self.xresolution = int(xresolution)     
        self.resolution = int(303)  #截面点数
        self.update1(a,b,n)
        self.sf=sf               #缩放控制系数
        self.yd=yd               #移动位置
        self.a0=a0              #中心线系数
        self.a1=a1
        self.a2=a2
        self.a3=a3
        self.a4=a4
        self.a5=a5

    def update(self,a,b,n):
        '''初始化参数'''
        self.x = np.linspace(0.,self.length,self.xresolution)   #0到length分成self.xresolution份
        self.a = np.ones([self.xresolution])*a    #1000个0.1
        self.b = np.ones([self.xresolution])*b
        self.n = np.ones([self.xresolution])*n

    def update1(self,a,b,n):
        '''初始化参数，与self.update不同在于给前缘进行加密'''
        headn = self.xresolution - int(0.95*self.xresolution)
        x = np.linspace(0., self.length, self.xresolution-headn)   #0到length分成self.xresolution份
        headx = np.linspace(x[0], x[1], headn+2)
        self.x = np.sort(np.append(x,headx[1:-1]))
        self.a = np.ones([self.xresolution])*a    #1000个0.1
        self.b = np.ones([self.xresolution])*b
        self.n = np.ones([self.xresolution])*n

    def yz_cal(self,aa,bb,nn,tt):      #计算每个截面的y，z值
        """ aa:z向半长轴、bb:y向半长轴、nn：次方数、tt：角度值（弧度、0-2pi）"""
        na = 2.0/nn
        sgn = np.zeros(tt.shape)
        mask = np.cos(tt)>0.
        sgn[mask] = 1.
        mask = np.cos(tt)<0.
        sgn[mask] = -1.
        z = (abs((cos(tt)))**na)*aa * sgn

        sgn = np.zeros(tt.shape)
        mask = np.sin(tt)>0.
        sgn[mask] = 1.
        mask = np.sin(tt)<0.
        sgn[mask] = -1.
        y = (abs((sin(tt)))**na)*bb * sgn
        return[z,y]
    
    def drawParameters(self):
        '''画出参数曲线'''
        
        x1d = (self.x[0:-1]+self.x[1:])/2.
        dx = self.x[1:]-self.x[0:-1]
        ax1 = plt.subplot(311)
        plt.plot(self.x,self.a, color='blue', label="Width")
        plt.grid()
        ax1d = ax1.twinx()
        plt.plot(x1d, (self.a[1:]-self.a[0:-1])/dx, color='red',ls='-.', label="Width")
        plt.title("Shape Parameters")
        plt.legend(loc=2)
        
        ax2 = plt.subplot(312, sharex=ax1)
        plt.plot(self.x,self.b,label="High")
        plt.grid()
        ax2d = ax2.twinx()
        plt.plot(x1d, (self.b[1:]-self.b[0:-1])/dx, color='red',ls='-.', label="High")
        plt.legend(loc=2)
        
        ax3 = plt.subplot(313, sharex=ax1)
        plt.plot(self.x, self.n, label="Shape")
        plt.grid()
        ax3d = ax3.twinx()
        plt.plot(x1d, (self.n[1:]-self.n[0:-1])/dx, color='red',ls='-.', label="Shape")
        plt.legend(loc=2)
        plt.tight_layout()
        plt.show()

    def applyFunc(self, myfunction,a,parameters, domainFuselage=[0.,1.], domainFunction=[0.,1.]):
        ''' 将某一个函数（myfuction）值叠加到某一个一维参数（a）上，应用的范围为domainFuselage，函数（myfunction）的域为domainFunction'''
        #从self.x 中取大domainFuselage中的x坐标
        mask = (self.x >=domainFuselage[0]) == (self.x < domainFuselage[1])
        x = self.x[mask]
        
        xmin = x.min()
        xmax = x.max()
        # 映射 x 到函数作用域
        xf = (x-xmin)/(xmax-xmin)*(domainFunction[1]-domainFunction[0])+domainFunction[0]
        parameters['x'] = xf
        y = myfunction(parameters)
        y=y+0.04*xf-0.04
        a[mask]+=y

    def applyCurve(self, curve,a,parameters, domainFuselage=[0.,1.]):#需要改的函数
        ''' 将某一个曲线叠加到参数a某一段上'''
        mask = (self.x >=domainFuselage[0]) == (self.x < domainFuselage[1])#取大于0小于1的值
        x = self.x[mask]
        X=((x-domainFuselage[0])/(domainFuselage[1]-domainFuselage[0]))
        y=k0+k1*X+k2*X**2+k3*X**3      #控制系数变化方程
        a[mask]+=y 

    def generateSurface(self):
        '''生成机身外形数据'''
        body = np.zeros([self.xresolution,self.resolution,3])
        t= np.linspace(0,2*np.pi,self.resolution)
        for i in range(self.xresolution):
            body[i,:,0] = self.x[i]
            xx=self.x[i]/self.length
            a = self.a[i]
            b = self.b[i]
            n = self.n[i]
            [z,y]=self.yz_cal(a,b,n,t)
            z=self.scal(z,y,b)*6
            y=self.move(y,xx)*6
            if self.x[i]<=4:
                y=y-0.025*self.x[i]**2+0.2*self.x[i]+0.3
            else:
                y=y+(0.4/7)*self.x[i]+0.7-1.6/7
            body[i,:,1],body[i,:,2] = z,y     
        
        self.body=body
        
    def generateTable(self,ka,kb,kn,domain=[4,5],doangle=[0.45*np.pi,0.55*np.pi]):
        '''生成凸台数据'''
        body = np.zeros([self.xresolution,self.resolution,3])
        t= np.linspace(0,2*np.pi,self.resolution)
        for i in range(self.xresolution):
            body[i,:,0] = self.x[i]
            xx=self.x[i]/self.length
            a = self.a[i]
            b = self.b[i]
            n = self.n[i]
            [z,y]=self.yz_cal(a,b,n,t)
            j=0
            if domain[0]<=self.x[i]<=domain[1]:
                for tt in np.arange(0,2*np.pi,2*np.pi/(self.resolution-1)):
                    if doangle[0]<=tt<=doangle[1]:
                        #凸台形状控制，改变对应的a,b,n
                        kx=(self.x[i]-domain[0])/(domain[1]-domain[0])
                        k=5.8*kx-14.4*(kx**2)+17.2*(kx**3)-8.6*(kx**4)+0.1
                        #k为过渡系数，控制凸台过渡形状
                        [aa,bb]=self.dab_cal(doangle[0],doangle[1],a,b,tt,k)
                        [z[j],y[j]]=self.yz_cal(ka*aa,kb*bb,kn*n,tt)
                    else:
                        pass
                    j=j+1 
                z=self.scal(z,y,b)*6
                y=self.move(y,xx)*6
                if self.x[i]<=4:
                    y=y-0.025*self.x[i]**2+0.2*self.x[i]+0.3
                else:
                    y=y+(0.4/7)*self.x[i]+0.7-1.6/7
                self.body[i,:,1],self.body[i,:,2] = z,y 
        
    def dab_cal(self,theta0,deltatheta,a,b,t,k):
        '''计算凸台的控制参数，a,b'''
        da = np.sqrt((np.cos((theta0-t)*np.pi/deltatheta)+1)/2)*0.2*a*k
        db = np.sqrt((np.cos((theta0-t)*np.pi/deltatheta)+1)/2)*0.2*b*k
        aa = a+da
        bb = b+db
        return [aa,bb] 

    def scal(self,z,y,b):
        '''通过sf（系数）控制缩放比例'''
        z=(((1-self.sf)/2/b)*y + (self.sf-1)/2 + 1)*z       #系数从1-sf变化
        return z

    def move(self,y,xx):
        '''中心线方程 控制移动位置'''
        x =xx
        y1=self.a0+self.a1*x+self.a2*x**2+self.a3*x**3+self.a4*x**4+self.a5*x**5
        y=y+y1
        return y
    
    def scala(self,a,k0=0,k1=0,k2=0,k3=0,domainFuselage=[0.,1.]):
        '''减小特定部位的参数值'''
        mask = (self.x >=domainFuselage[0]) == (self.x < domainFuselage[1])#取大于0小于1的值
        x = self.x[mask]
        X=((x-domainFuselage[0])/(domainFuselage[1]-domainFuselage[0]))
        y=k0+k1*X+k2*X**2+k3*X**3      #控制系数变化方程
        a[mask]+=y     

def naca00(parameters):    
    #x是为值为0-1的一维数组，tt代表厚度
    x = parameters['x']
    tt = parameters['tt']
    tt = tt*0.01

    #xx=(x-0)/11
    #tt=0.74861*xx**3-1.40799*xx**2+0.65982*xx        #naca0012厚度曲线
    
    y = (tt/0.2)*(0.2969*np.sqrt(x) - 0.126*x - 0.3516*x*x + 0.2843*x*x*x -0.1015*x*x*x*x)
    return y





## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

