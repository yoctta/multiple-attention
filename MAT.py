import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.xception import xception
from models.efficientnet2 import EfficientNet as EfficientNet2
from models.efficientnet import EfficientNet
from utils import cont_grad
import kornia
class AttentionMap(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AttentionMap, self).__init__()
        self.register_buffer('mask',torch.zeros([1,1,24,24]))
        self.mask[0,0,2:-2,2:-2]=1
        self.num_attentions=out_channels
        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1) #extracting feature map from backbone
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        if self.num_attentions==0:
            return torch.ones([x.shape[0],1,1,1],device=x.device)
        x = self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)+1
        mask=F.interpolate(self.mask,(x.shape[2],x.shape[3]),mode='nearest')
        return x*mask


class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, features, attentions,norm=2):
        H, W = features.size()[-2:]
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions=F.interpolate(attentions,size=(H,W), mode='bilinear', align_corners=True)
        if norm==1:
            attentions=attentions+1e-8
        if len(features.shape)==4:
            feature_matrix=torch.einsum('imjk,injk->imn', attentions, features)
        else:
            feature_matrix=torch.einsum('imjk,imnjk->imn', attentions, features)
        if norm==1:
            w=torch.sum(attentions,dim=(2,3)).unsqueeze(-1)
            feature_matrix/=w
        if norm==2:
            feature_matrix = F.normalize(feature_matrix,p=2,dim=-1)
        if norm==3:
            w=torch.sum(attentions,dim=(2,3)).unsqueeze(-1)+1e-8
            feature_matrix/=w
        return feature_matrix



class Texture_Enhance(nn.Module):
    def __init__(self,num_features,num_attentions):
        super().__init__()
        self.output_features=num_features
        self.conv_extract=nn.Conv2d(num_features,num_features,3,padding=1)
        self.conv0=nn.Conv2d(num_features*num_attentions,num_features*num_attentions,5,padding=2,groups=num_attentions)
        self.conv1=nn.Conv2d(num_features*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn1=nn.BatchNorm2d(num_features*num_attentions)
        self.conv2=nn.Conv2d(num_features*2*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn2=nn.BatchNorm2d(2*num_features*num_attentions)
        self.conv3=nn.Conv2d(num_features*3*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn3=nn.BatchNorm2d(3*num_features*num_attentions)
        self.conv_last=nn.Conv2d(num_features*4*num_attentions,num_features*num_attentions,1,groups=num_attentions)
        self.bn4=nn.BatchNorm2d(4*num_features*num_attentions)
        self.bn_last=nn.BatchNorm2d(num_features*num_attentions)
        self.conv_d=nn.Sequential(nn.Conv2d(num_features,num_features//2,5,padding=2),nn.BatchNorm2d(num_features//2),nn.ReLU(inplace=True),nn.Conv2d(num_features//2,num_features,1))
        self.M=num_attentions
    def cat(self,a,b):
        B,C,H,W=a.shape
        c=torch.cat([a.reshape(B,self.M,-1,H,W),b.reshape(B,self.M,-1,H,W)],dim=2).reshape(B,-1,H,W)
        return c

    def forward(self,feature_maps,attention_maps):
        feature_maps=self.conv_extract(feature_maps)
        attention_size=(attention_maps.shape[2],attention_maps.shape[3])
        feature_maps_d=F.adaptive_avg_pool2d(feature_maps,attention_size)
        if feature_maps.size(2)>feature_maps_d.size(2):
            feature_maps=feature_maps-F.interpolate(feature_maps_d,(feature_maps.shape[2],feature_maps.shape[3]),mode='nearest')
        feature_maps_d=self.conv_d(cont_grad(feature_maps_d))
        B,N,H,W=feature_maps.shape
        attention_maps=F.interpolate(attention_maps,(H,W),mode='bilinear',align_corners=True)
        attention_maps=(F.tanh(attention_maps)).unsqueeze(2)
        feature_maps=feature_maps.unsqueeze(1)
        feature_maps=(feature_maps*attention_maps).reshape(B,-1,H,W)
        feature_maps0=self.conv0(feature_maps)
        feature_maps1=self.conv1(F.relu(self.bn1(feature_maps0),inplace=True))
        feature_maps1_=self.cat(feature_maps0,feature_maps1)
        feature_maps2=self.conv2(F.relu(self.bn2(feature_maps1_),inplace=True))
        feature_maps2_=self.cat(feature_maps1_,feature_maps2)
        feature_maps3=self.conv3(F.relu(self.bn3(feature_maps2_),inplace=True))
        feature_maps3_=self.cat(feature_maps2_,feature_maps3)
        feature_maps=F.relu(self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_),inplace=True))),inplace=True)
        feature_maps=feature_maps.reshape(B,-1,N,H,W)
        return feature_maps,feature_maps_d


class Auxiliary_Loss(nn.Module):
    def __init__(self,M,N,C,alpha=0.05,margin=1,inner_margin=[0.01,0.02]):
        super().__init__()
        self.register_buffer('feature_centers',torch.zeros(M,N))
        self.register_buffer('alpha',torch.tensor(alpha))
        self.num_classes=C
        self.margin=margin
        self.atp=AttentionPooling()
        self.register_buffer('inner_margin',torch.Tensor(inner_margin))
    def forward(self,feature_matrix,feature_map_d,attentions,y):
        B,M,N=feature_matrix.size()
        feature_centers=self.feature_centers.detach()
        center_momentum=feature_matrix-feature_centers
        fcts=self.alpha*torch.mean(center_momentum,dim=0)+feature_centers
        fctsd=fcts.detach()
        if self.training:
            with torch.no_grad():
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(fctsd,torch.distributed.ReduceOp.SUM)
                    fctsd/=torch.distributed.get_world_size()
                self.feature_centers=fctsd  
        inner_margin=torch.gather(self.inner_margin.repeat(B,1),1,y.unsqueeze(1))
        #diffs=torch.sum((feature_map_d.unsqueeze(1)-fcts.view(1,M,N,1,1))**2,dim=2,keepdim=True)
        #diffs=self.atp(diffs,attentions).unsqueeze(-1)
        #intra_class_loss=F.relu(diffs-inner_margin)
        #intra_class_loss=torch.mean(intra_class_loss)
        intra_class_loss=F.relu(torch.sqrt(torch.sum((feature_matrix-fcts)**2,dim=-1))-inner_margin)
        intra_class_loss=torch.mean(intra_class_loss)
        inter_class_loss=0
        for j in range(M):
            for k in range(j+1,M):
                inter_class_loss+=F.relu(self.margin-torch.dist(fcts[j],fcts[k]),inplace=False)
        return intra_class_loss/B+inter_class_loss*5


class MAT(nn.Module):
    def __init__(self, net='xception',feature_layer='b3',attention_layer='final',num_classes=2, M=8,mid_dims=256,\
    dropout_rate=0.5,drop_final_rate=0.5, pretrained=False,alpha=0.05,size=(380,380),margin=1,inner_margin=[0.01,0.02]):
        super(MAT, self).__init__()
        self.num_classes = num_classes
        self.M = M
        if 'xception' in net:
            self.net=xception(num_classes)
        elif net.split('-')[0]=='efficientnet':
            self.net=EfficientNet.from_pretrained(net,advprop=True, num_classes=num_classes)
        elif net.split('-')[0]=='efficientnet2':
            net=net.replace('efficientnet2','efficientnet')
            self.net=EfficientNet2.from_pretrained(net,advprop=True, num_classes=num_classes,feature_layer=feature_layer)
        self.feature_layer=feature_layer
        self.attention_layer=attention_layer
        with torch.no_grad():
            layers = self.net(torch.zeros(1,3,size[0],size[1]))
        num_features=layers[self.feature_layer].shape[1]
        self.mid_dims=mid_dims
        if pretrained:
            a=torch.load(pretrained,map_location='cpu')
            keys={i:a['state_dict'][i] for i in a.keys() if i.startswith('net')}
            if not keys:
                keys=a['state_dict']
            self.net.load_state_dict(keys,strict=False)
        self.attentions = AttentionMap(layers[self.attention_layer].shape[1], self.M)
        self.atp=AttentionPooling()
        self.texture_enhance=Texture_Enhance(num_features,M)
        self.num_features=self.texture_enhance.output_features
        self.projection_local=nn.Sequential(nn.Linear(M*self.num_features,mid_dims),nn.Hardswish(),nn.Linear(mid_dims,mid_dims))
        self.project_final=nn.Linear(layers['final'].shape[1],mid_dims)
        self.ensemble_classifier_fc=nn.Sequential(nn.Linear(mid_dims*2,mid_dims),nn.Hardswish(),nn.Linear(mid_dims,num_classes))
        self.auxiliary_loss=Auxiliary_Loss(M,num_features,num_classes,alpha,margin,inner_margin)
        self.dropout=nn.Dropout2d(dropout_rate,inplace=True)
        self.dropout_final=nn.Dropout(drop_final_rate,inplace=True)

    def train_batch(self,x,y,jump_aux=False,drop_final=False):
        layers = self.net(x)
        if self.feature_layer=='logits':
            logits=layers['logits']
            loss=F.cross_entropy(logits,y)
            return dict(loss=loss,logits=logits)
        feature_maps = layers[self.feature_layer]
        raw_attentions = layers[self.attention_layer]
        attention_maps_=self.attentions(raw_attentions)
        dropout_mask=self.dropout(torch.ones([attention_maps_.shape[0],self.M,1],device=x.device))
        attention_maps=attention_maps_*torch.unsqueeze(dropout_mask,-1)
        feature_maps,feature_maps_d=self.texture_enhance(feature_maps,attention_maps)
        feature_matrix_d=self.atp(feature_maps_d,attention_maps_)
        feature_matrix_=self.atp(feature_maps,attention_maps_)
        feature_matrix=feature_matrix_*dropout_mask

        B,M,N = feature_matrix.size()
        if not jump_aux:
            aux_loss=self.auxiliary_loss(feature_matrix_d,feature_maps_d,attention_maps_,y)
        else:
            aux_loss=0
        feature_matrix=feature_matrix.view(B,-1)
        feature_matrix=F.hardswish(self.projection_local(feature_matrix))
        final=layers['final']
        attention_maps=attention_maps.sum(dim=1,keepdim=True)
        final=self.atp(final,attention_maps,norm=1).squeeze(1)
        final=self.dropout_final(final)
        projected_final=F.hardswish(self.project_final(final))
        projected_final=self.dropout(projected_final.view(B,1,-1)).view(B,-1)
        if drop_final:
            projected_final*=0
        feature_matrix=torch.cat((feature_matrix,projected_final),1)
        ensemble_logit=self.ensemble_classifier_fc(feature_matrix)
        ensemble_loss=F.cross_entropy(ensemble_logit,y)
        return dict(ensemble_loss=ensemble_loss,aux_loss=aux_loss,attention_maps=attention_maps_,ensemble_logit=ensemble_logit,feature_matrix=feature_matrix_,feature_matrix_d=feature_matrix_d)


    def forward(self, x,y=0,train_batch=False,AG=None):
        if train_batch:
            if AG is None:
                return self.train_batch(x,y)
            else:
                loss_pack=self.train_batch(x,y)
                with torch.no_grad():
                    Xaug,index=AG.agda(x,loss_pack['attention_maps'])
                loss_pack2=self.train_batch(Xaug,y,jump_aux=True)
                loss_pack['AGDA_ensemble_loss']=loss_pack2['ensemble_loss']
                loss_pack['AGDA_aux_loss']=loss_pack2['aux_loss']
                one_hot=F.one_hot(index,self.M)
                loss_pack['match_loss']=torch.mean(torch.norm(loss_pack2['feature_matrix_d']-loss_pack['feature_matrix_d'],dim=-1)*(torch.ones_like(one_hot)-one_hot))

                return loss_pack
        layers = self.net(x)
        if self.feature_layer=='logits':
            logits=layers['logits']
            return logits
        raw_attentions = layers[self.attention_layer]
        attention_maps=self.attentions(raw_attentions)
        feature_maps = layers[self.feature_layer]
        feature_maps=self.texture_enhance(feature_maps,attention_maps)[0]
        feature_matrix=self.atp(feature_maps,attention_maps)
        B,M,N = feature_matrix.size()
        feature_matrix=self.dropout(feature_matrix)
        feature_matrix=feature_matrix.view(B,-1)
        feature_matrix=F.hardswish(self.projection_local(feature_matrix))
        final=layers['final']
        attention_maps2=attention_maps.sum(dim=1,keepdim=True)
        final=self.atp(final,attention_maps2).squeeze(1)
        projected_final=F.hardswish(self.project_final(final))
        feature_matrix=torch.cat((feature_matrix,projected_final),1)
        ensemble_logit=self.ensemble_classifier_fc(feature_matrix)
        return ensemble_logit

def load_state(net,ckpt):
    sd=net.state_dict()
    nd={}
    for i in ckpt:
        if i in sd and sd[i].shape==ckpt[i].shape:
            nd[i]=ckpt[i]
    net.load_state_dict(nd,strict=False)

class netrunc(nn.Module):
    def __init__(self, net='xception',feature_layer='b3',num_classes=2,dropout_rate=0.5, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        if 'xception' in net:
            self.net=xception(num_classes,escape=feature_layer)
        elif net.split('-')[0]=='efficientnet':
            self.net=EfficientNet.from_pretrained(net,advprop=True, num_classes=num_classes,escape=feature_layer)
        self.feature_layer=feature_layer
        with torch.no_grad():
            layers = self.net(torch.zeros(1,3,100,100))
        num_features=layers[self.feature_layer].shape[1]
        if pretrained:
            a=torch.load(pretrained,map_location='cpu')
            keys={i:a['state_dict'][i] for i in a.keys() if i.startswith('net')}
            if not keys:
                keys=a['state_dict']
            load_state(self.net,keys)
        self.pooling=nn.AdaptiveAvgPool2d(1)
        self.texture_enhance=Texture_Enhance(num_features)
        self.num_features=self.texture_enhance.output_features
        self.fc=nn.Linear(self.num_features,self.num_classes)
        self.dropout=nn.Dropout(dropout_rate)


    def forward(self, x):
        layers = self.net(x)
        feature_maps = layers[self.feature_layer]
        feature_maps=self.texture_enhance(feature_maps,(feature_maps.shape[2]//4,feature_maps.shape[3]//4))[0]
        x=self.pooling(feature_maps)
        x = x.flatten(start_dim=1)
        x=self.dropout(x)
        x=self.fc(x)
        return x
