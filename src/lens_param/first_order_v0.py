import torch
import matplotlib.pyplot as plt


def ray_trace(r0,drdz0):
    f=5
    dz=0.1
    optic1=torch.tensor([[1,0],[-1/f,1]],dtype=float)
    optic1_loc=2
    dz_lens=torch.tensor([[1,dz],[0,1]],dtype=float)
    r=[r0]
    drdz=[drdz0]
    z=[0]
    while z[-1]<10:
        if z[-1]>=optic1_loc and z[-1]<optic1_loc+dz:
            r_vec=torch.mv(optic1,torch.tensor([r[-1],drdz[-1]],dtype=float))
            r.append(r_vec[0])
            drdz.append(r_vec[1])
        else:
            r_vec=torch.mv(dz_lens,torch.tensor([r[-1],drdz[-1]],dtype=float))
            r.append(r_vec[0])
            drdz.append(r_vec[1])
        z.append(z[-1]+dz)
    return torch.tensor(r),torch.tensor(drdz),torch.tensor(z)

n=10
r0=torch.linspace(-2,2,n)
for i in range(n):
    r,drdz,z=ray_trace(r0[i],0)
    plt.plot(z,r)

#create normal tracing ray matrix
#fill in matrix at point of using a function
#run matrix to get set of lines


plt.show()
