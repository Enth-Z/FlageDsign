from numpy import linspace,logspace,zeros,pi,cos,sin,pi

class SurfacePart ():
    """面部件"""
    def __init__(self,imax=0,jmax=0,name="UnnamedSurfacePart"):
        """imax，jmax为两个方向的尺寸。"""
        self.imax = imax
        self.jmax = jmax
        self.xyz = zeros([self.imax,self.jmax,3])
        self.name = name

    def exporttxt(self, fn):
        """按照i条输出，文件名fn"""
        of = open(fn,'w')
        of.write("# %s imax = %i jmax = %i\n"%(self.name,self.imax,self.jmax))
        for i in self.xyz:
            for j in i:
                of.write("%f %f %f\n"%(j[0],j[1],j[2]))
            of.write('\n')
        of.close()


class RotPart(SurfacePart):
    """旋成体"""
    def __init__ (self,nlines,npoints):
        SurfacePart.__init__(self,imax=nlines,jmax=npoints)
        self.nlines = nlines
        self.resolution = npoints

