import torch
import random
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import kornia

class AGDA(nn.Module):
    def __init__(self,kernel_size=7,dilation=2,sigma=5,threshold=(0.4,0.6),zoom=(3,5),scale_factor=0.5,noise_rate=0.1,mode='soft'):
        super().__init__()
        self.kernel_size=kernel_size
        self.dilation=dilation
        self.sigma=sigma
        self.noise_rate=noise_rate
        self.scale_factor=scale_factor
        self.threshold=threshold
        self.zoom=zoom
        self.mode=mode
        self.filter=kornia.filters.GaussianBlur2d((self.kernel_size,self.kernel_size),(self.sigma,self.sigma))
        if mode=='pointwise':
            distmap=(np.arange(kernel_size)-(kernel_size-1)/2)**2
            distmap=distmap.reshape(1,-1)+distmap.reshape(-1,1)
            distmap*=dilation**2
            rec={}
            for i in range(kernel_size):
                for j in range(kernel_size):
                    d=distmap[i,j]
                    if d not in rec:
                        rec[d]=[(i,j)]
                    else:
                        rec[d].append((i,j))
            dist=np.array(list(rec.keys()))
            mod=np.zeros([dist.shape[0],kernel_size,kernel_size])
            ct=[]
            for i in range(dist.shape[0]):
                ct.append(len(rec[dist[i]]))
                for j in rec[dist[i]]:
                    mod[i,j[0],j[1]]=1
            mod=torch.from_numpy(mod).reshape(-1,1,kernel_size,kernel_size).repeat([3,1,1,1]).float()
            self.register_buffer('kernel',mod)
            self.register_buffer('dist',torch.from_numpy(dist).reshape(1,-1,1,1).float())
            self.register_buffer('ct',torch.Tensor(ct).reshape(1,-1,1,1).float())


    def soft_drop(self,X,attention_map):
        with torch.no_grad():
            B,C,H,W=X.size()
            X=F.pad(X,tuple([int((self.kernel_size-1)/2*self.dilation)]*4),mode='reflect')
            X=F.conv2d(X,self.kernel,dilation=self.dilation,groups=3).view(B,3,-1,H,W)
            sigma=attention_map*self.sigma
            ex=torch.exp(-1.0/(2*sigma)).view(B,1,H,W)
            ex=ex**self.dist
            c=torch.sum(self.ct*ex,dim=1,keepdim=True)
            X=torch.einsum('bijmn,bjmn->bimn',X,ex)
            X/=c
        return X

    def mod_func(self,x):
        if type(self.threshold)==tuple:
                thres=random.uniform(*self.threshold)
        else:
            thres=self.threshold
        if type(self.zoom)==tuple:
                zoom=random.uniform(*self.zoom)
        else:
            zoom=self.zoom
        bottom=torch.sigmoid((torch.tensor(0.)-thres)*zoom)
        return (torch.sigmoid((x-thres)*zoom)-bottom)/(1-bottom)

    def soft_drop2(self,x,attention_map):
        with torch.no_grad():
            attention_map=self.mod_func(attention_map)
            B,C,H,W=x.size()
            xs=F.interpolate(x,scale_factor=self.scale_factor,mode='bilinear',align_corners=True)
            xs=self.filter(xs)
            xs+=torch.randn_like(xs)*self.noise_rate
            xs=F.interpolate(xs,(H,W),mode='bilinear',align_corners=True)
            x=x*(1-attention_map)+xs*attention_map
        return x


    def hard_drop(self,X,attention_map):
        with torch.no_grad():
            if type(self.threshold)==tuple:
                thres=random.uniform(*self.threshold)
            else:
                thres=self.threshold
            attention_mask=attention_map<thres
            X=attention_mask.float()*X
        return X
    
    def mix_drop(self,X,attention_map):
        if random.randint(0,1)==0:
            return self.hard_drop(X,attention_map)
        else:
            return self.soft_drop2(X,attention_map)

    def agda(self,X,attention_map):
        with torch.no_grad():
            attention_weight=torch.sum(attention_map,dim=(2,3))
            attention_map=F.interpolate(attention_map,(X.size(2),X.size(3)),mode="bilinear",align_corners=True)
            attention_weight=torch.sqrt(attention_weight+1)
            index=torch.distributions.categorical.Categorical(attention_weight).sample()
            index1=index.view(-1,1,1,1).repeat(1,1,X.size(2),X.size(3))
            attention_map=torch.gather(attention_map,1,index1)
            atten_max=torch.max(attention_map.view(attention_map.shape[0],1,-1),2)[0]+1e-8
            attention_map=attention_map/atten_max.view(attention_map.shape[0],1,1,1)
            if self.mode=='soft':
                return self.soft_drop2(X,attention_map),index
            elif self.mode=='pointwise':
                return self.soft_drop(X,attention_map),index
            elif self.mode=='hard':
                return self.hard_drop(X,attention_map),index
            elif self.mode=='mix':
                return self.mix_drop(X,attention_map),index
                





    