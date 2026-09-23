import torch
import pyvista as pv
import matplotlib.pyplot as plt
from time import time


class optical_system:
    #create normal tracing ray matrix
    def __init__(self,z_length=1,n=1):
        dz=z_length/n
        z=torch.linspace(0,z_length,n+1)
        M=torch.broadcast_to(torch.tensor([[1,dz,0,0,0],[0,1,0,0,0],[0,0,1,dz,0],[0,0,0,1,0],[0,0,0,0,1]],requires_grad=True),(n,5,5))
        self.n=n
        self.z=z
        self.M=M

    #fill in matrix at point of lens
    def add_lens(self,z_lens,f_x,f_y):
        """input:   z_lens -> z position of lens on z_axis | f_x,f_y -> focus of lens in x and y direction respectively"""
        bool=(self.z>=z_lens)
        arg_before=torch.argwhere(bool)[0]-1

        

        component=torch.tensor([[1,0,0,0,0],[-1/f_x,1,0,0,0],[0,0,1,0,0],[0,0,-1/f_y,1,0],[0,0,0,0,1]])
      
        M_old=self.M.clone() 
        z_old=self.z.clone()
        M_new=torch.zeros(self.n+2,5,5)
        z_new=torch.zeros(self.n+3)
        for i in range(self.n+2):
            if i<arg_before:
                M_new[i]=M_old[i]
                z_new[i]=z_old[i]
            elif i==arg_before:
                dz=z_lens-z_old[arg_before]
                M_new[i]=torch.tensor([[1,dz,0,0,0],[0,1,0,0,0],[0,0,1,dz,0],[0,0,0,1,0],[0,0,0,0,1]])
                z_new[i]=z_old[i]
            elif i==arg_before+1:
                M_new[i]=component
                z_new[i]=z_new[i-1]+dz
            elif i==arg_before+2:
                dz=z_old[arg_before+1]-z_lens
                M_new[i]=torch.tensor([[1,dz,0,0,0],[0,1,0,0,0],[0,0,1,dz,0],[0,0,0,1,0],[0,0,0,0,1]])
                z_new[i]=z_new[i-1]
            elif i>arg_before+2:
                M_new[i]=M_old[i-2]
                z_new[i]=z_old[i-2]
            else:
                print("something went wrong")
        z_new[-1]=z_old[-1]

        self.M=M_new
        self.z=z_new
        self.n+=2

    def run_ray(self,r0=(0,0),drdz0=(0,0)):
        M=self.M.clone()

        x=[r0[0]]
        dxdz=[drdz0[0]]

        y=[r0[1]]
        dydz=[drdz0[1]]
        for m in range(self.n):
            r_vec=torch.mv(M[m],torch.tensor([x[-1],dxdz[-1],y[-1],dydz[-1],1]))
            x.append(r_vec[0]),dxdz.append(r_vec[1]),y.append(r_vec[2]),dydz.append(r_vec[3])

        return self.z, torch.tensor(x), torch.tensor(dxdz), torch.tensor(y), torch.tensor(dydz)

    def run_pixel(self,r0,n_raysppa,intensity,drdz_range):
        r1=[]
        for theta in torch.linspace(-drdz_range/2,drdz_range/2,n_raysppa+2)[1:-1]:
            for phi in torch.linspace(-drdz_range/2,drdz_range/2,n_raysppa+2)[1:-1]:
                drdz0=(torch.tan(theta),torch.tan(phi))
                _,x,_,y,_=self.run_ray(r0,drdz0)
                r1.append(torch.stack([x[-1],y[-1],intensity/n_raysppa]))
                
        r1=torch.stack(r1)
        return r1



    def run_object(self,object,n_raysppa=10,grid_l=(4,4),grid_p=(64,64),theta_range=torch.pi/2):
        """this assumes homogenous radial dispersion of intensity"""
        x=torch.linspace(-grid_l[0]/2,grid_l[0]/2,grid_p[0])
        y=torch.linspace(-grid_l[1]/2,grid_l[1]/2,grid_p[1])
        X,Y=torch.meshgrid(x,y,indexing="ij")
        r0_list=[]
        r1_list=[]
        print("\n started with ray tracing each grid point:")
        counter=0
        countermax=grid_p[0]*grid_p[1]
        for i in range(len(x)):
            for j in range(len(y)):
                intensity=object(torch.sqrt(X[i,j]**2+Y[i,j]**2),torch.arctan2(X[i,j],Y[i,j]))
                if intensity != 0:
                    r0=torch.stack([X[i,j],Y[i,j],intensity])
                    r1=self.run_pixel(r0[0:2] ,n_raysppa,intensity ,theta_range) #r1 is list of landing points with intensities
                    r0_list.append(r0)
                    r1_list.append(r1)
                counter+=1
                print("\r %i procent simulated" %(int(100*counter/countermax)),end="")
                

        r0_list=torch.stack(r0_list)
        r1_list=torch.cat(r1_list)
        print("\n showing construction...")
        fig, axs=plt.subplots(1,2)
        axs[0].hist2d(r0_list[:,0],r0_list[:,1],bins=grid_p,weights=r0_list[:,2],)
        axs[1].hist2d(r1_list[:,0],r1_list[:,1],bins=grid_p,weights=r1_list[:,2])
        axs[0].set_title("Object"),axs[0].set_xlabel("x postition"), axs[0].set_ylabel("y postition")
        axs[1].set_title("Image"),axs[1].set_xlabel("x postition"), axs[1].set_ylabel("y postition")
        plt.show()

    def plot_simple(self,n_rays):
        theta0_list=torch.linspace(0,2*torch.pi,n_rays+1)[:-1]
        r=1
        plotter=pv.Plotter()
        for theta0 in theta0_list:
            z,x,_,y,_=self.run_ray(r0=(r*torch.cos(theta0),r*torch.sin(theta0)))
            for i in range(len(z)-1):
                p_from=(x[i],y[i],z[i])
                p_to=(x[i+1],y[i+1],z[i+1])
                if z[i]!=z[i+1]:
                    ray=pv.Line(p_from,p_to)
                    ray["theta"]=[theta0,theta0]
                    plotter.add_mesh(ray,line_width=3,scalars="theta",cmap="plasma")
        plotter.show()



#define optical system
opt=optical_system(z_length=20)
opt.add_lens(10,5,5) #z position of lens, focus in x, focus in yopt.add_lens(8,-5,-5)
#opt.add_lens(10,5,4.5)
#opt.add_lens(10,5,4.5)

opt.plot_simple(10)

#define object
def I_object(r,theta,r0=1,d=0.1):
    if (r>=r0-d/2)&(r<r0+d/2):
        return torch.tensor(1)
    else:
        return torch.tensor(0)

#opt.run_object(I_object,n_raysppa=10,grid_l=(4,4),grid_p=(128,128),theta_range=torch.pi/2)