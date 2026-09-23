import torch
import pyvista as pv
import matplotlib.pyplot as plt


class optical_system:
    #create normal tracing ray matrix
    def __init__(self,n,z_length):
        dz=z_length/(n-1)
        z=torch.linspace(0,z_length,n)
        print(z)
        M=torch.broadcast_to(torch.tensor([[1,dz,0,0,0],[0,1,0,0,0],[0,0,1,dz,0],[0,0,0,1,0],[0,0,0,0,1]],requires_grad=True),(n,5,5))
        self.n=n
        self.z=z
        self.M=M
        self.dz=dz

    #fill in matrix at point of lens
    def add_lens(self,z_lens,f_x,f_y):
        bool=(self.z>=z_lens)&(self.z<z_lens+self.dz)

        component=torch.tensor([[1,0,0,0,0],[-1/f_x,1,0,0,0],[0,0,1,0,0],[0,0,-1/f_y,1,0],[0,0,0,0,1]])
      
        M_internal=self.M.clone() #required to use boolean
        original=M_internal[bool][0].clone()

        M_internal[bool]=torch.mm(original,component)
        self.M=M_internal

    def run(self,r0=(0,0),drdz0=(0,0)):
        M=self.M.clone()

        x=[r0[0]]
        dxdz=[drdz0[0]]

        y=[r0[1]]
        dydz=[drdz0[1]]
        for m in range(self.n-1):
            r_vec=torch.mv(M[m],torch.tensor([x[-1],dxdz[-1],y[-1],dydz[-1],1]))
            x.append(r_vec[0]),dxdz.append(r_vec[1]),y.append(r_vec[2]),dydz.append(r_vec[3])
        return self.z, torch.tensor(x), torch.tensor(dxdz), torch.tensor(y), torch.tensor(dydz)


    def plot(self,n_rays):
        theta0_list=torch.linspace(0,2*torch.pi,n_rays+1)[:-1]
        r=1
        print(theta0_list)
        #plotter=pv.Plotter()
        for theta0 in theta0_list:
            z,x,_,y,_=self.run(r0=(r*torch.cos(theta0),r*torch.sin(theta0)))
            plt.plot(z,x)
        plt.show()



opt=optical_system(100,12)
opt.add_lens(5,5,5)

M=opt.M.clone()
print(M)
#torch.autograd(M)

opt.plot(6)

