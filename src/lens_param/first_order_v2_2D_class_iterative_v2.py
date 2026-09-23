import torch
import matplotlib.pyplot as plt


class optical_system:
    #create normal tracing ray matrix
    def __init__(self,n,z_length):
        dz=z_length/(n-1)
        z=torch.linspace(0,z_length,n)
        M=torch.broadcast_to(torch.tensor([[1,dz],[0,1]],dtype=torch.float),(n,2,2))
        self.n=n
        self.z=z
        self.M=M
        self.dz=dz
        self.app=torch.ones_like(z)*torch.inf

    #fill in matrix at point of lens
    def add_lens(self,z_lens,f_lens):
        bool=(self.z>=z_lens)&(self.z<z_lens+self.dz)
        if type(f_lens)==float or type(f_lens)==int:
            f_lens=torch.tensor(f_lens,dtype=torch.float)

        component=torch.stack([torch.tensor([1,0]),torch.stack([-1/f_lens,torch.tensor(1)])])
        
        M_internal=self.M.clone() #required to use boolean
        original=M_internal[bool][0].clone()

        M_internal[bool]=torch.mm(original,component)
        self.M=M_internal

    def add_aperture(self,z_aperture,radius):
        bool=(self.z>=z_aperture)&(self.z<z_aperture+self.dz)
        self.app[bool]=radius
        


    def run(self,r0,drdz0):
        M=self.M.clone()
        app=self.app.clone()
        

        r=[torch.as_tensor(r0,dtype=torch.float)]
        drdz=[torch.as_tensor(drdz0,dtype=torch.float)]
        for m in range(self.n-1):
            if torch.abs(r[-1])>app[m]:
                return self.z[:m+1],r,drdz
            r_vec=torch.mv(M[m],torch.stack([r[-1],drdz[-1]]))
            r.append(r_vec[0]),drdz.append(r_vec[1])
        return self.z, r, drdz

    def rays_init_cone(self,cone_angle,n):
        if type(cone_angle)==float or type(cone_angle)==int:
                    cone_angle=torch.tensor(cone_angle,dtype=torch.float)
        drdz_range=torch.linspace(-torch.tan(torch.pi*cone_angle/180),torch.tan(torch.pi*cone_angle/180),n)
        r_range=torch.zeros_like(drdz_range)
        self.rays=torch.stack((r_range,drdz_range),dim=1)

    def rays_init_flat(self,radius,n):
        r_range=torch.linspace(-radius,radius,n)
        drdz_range=torch.zeros_like(r_range)
        self.rays=torch.stack((r_range,drdz_range),dim=1)


    def plot(self,size):
        rays=self.rays.clone()
        for i in range(rays.shape[0]):
            z,r,_=self.run(rays[i][0],rays[i][1])
            plt.plot(z,r,"g")
        plt.ylim(-size/2,size/2)
        plt.show()

    def optimize(self,r_desired,f_guess=4,n_iterations=50,patience=10,lr=1):
        loss_list=[]
        f_list=[]
        lr_list=[]
        f_guess=torch.tensor(f_guess,requires_grad=True,dtype=torch.float) #initial guess


        optimizer =torch.optim.Adam([f_guess],lr=lr)
        #optimizer =torch.optim.Adam([f_guess],lr=1)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=patience)
        for i in range(n_iterations):
            optimizer.zero_grad()
            #create setup and find r at measurement screen for z=12
            self.__init__(100,12)
            self.add_lens(5,f_guess)
            z,r,drdz=self.run(r0=10,drdz0=0)
            r_found=r[-1]

            #calculate loss:
            loss=(r_found-r_desired)**2
            loss_list.append(loss.item())
            f_list.append(f_guess.item())
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            lr=optimizer.param_groups[0]["lr"]
            lr_list.append(lr)

            print("iteration %i, loss = %.2f, f_guess= %.2f"%(i,loss.item(),f_guess.item()))

        plt.plot(loss_list,".")
        plt.plot(lr_list)
        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel("loss/learning rate")
        plt.show()



opt=optical_system(100,2)
opt.rays_init_cone(15,100)

### optical configuration -- SEM -- 2 condenser setup
opt.add_lens(0.2,0.2)
opt.add_aperture(0.6,0.03)
opt.add_lens(0.8,0.07)
opt.add_aperture(1.3,0.05)
opt.add_lens(1.5,0.14)
opt.plot(0.5)



### optimize to this parameter using ADAM/SGD + lr_scheduler -> uses: opt_check=optical_system(100,12) + single lens system
opt.optimize(-6.450220584869385,5,lr=0.01,patience=100)   #aims to get f=4.2



