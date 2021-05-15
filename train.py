import os
import time
import logging
import warnings
import numpy 
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from models.MAT import MAT
from datasets.dataset import DeepfakeDataset
from AGDA import AGDA
import cv2
from utils import dist_average,ACC
#from torch.utils.tensorboard import SummaryWriter
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# GPU settings
assert torch.cuda.is_available()
#torch.autograd.set_detect_anomaly(True)
def load_state(net,ckpt):
    sd=net.state_dict()
    nd={}
    goodmatch=True
    for i in ckpt:
        if i in sd and sd[i].shape==ckpt[i].shape:
            nd[i]=ckpt[i]
            #print(i)
        else:
            print('fail to load %s'%i)
            goodmatch=False
    net.load_state_dict(nd,strict=False)
    return goodmatch
def main_worker(local_rank,world_size,rank_offset,config):
    rank=local_rank+rank_offset
    if rank==0:
        logging.basicConfig(
        filename=os.path.join('runs', config.name,'train.log'),
        filemode='a',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")
    dist.init_process_group(backend='nccl', init_method=config.url,world_size=world_size, rank=rank)
    # if rank==0:
    #     try:
    #         os.remove('/tmp/.pytorch_distribute')
    #     except:
    #         pass
    numpy.random.seed(1234567)
    torch.manual_seed(1234567)
    torch.cuda.manual_seed(1234567)
    torch.cuda.set_device(local_rank)
    train_dataset = DeepfakeDataset(phase='train',**config.train_dataset)
    validate_dataset=DeepfakeDataset(phase='test',**config.val_dataset)
    train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
    validate_sampler=torch.utils.data.distributed.DistributedSampler(validate_dataset)
    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,sampler=train_sampler,pin_memory=True,num_workers=config.workers)
    validate_loader=torch.utils.data.DataLoader(validate_dataset, batch_size=config.batch_size,sampler=validate_sampler,pin_memory=True,num_workers=config.workers)
    logs = {}
    start_epoch = 0
    net = MAT(**config.net_config)
    for i in config.freeze:
        if 'backbone' in i:
            net.net.requires_grad_(False)
        elif 'attention' in i:
            net.attentions.requires_grad_(False)
        elif 'feature_center' in i:
            net.auxiliary_loss.alpha=0
        elif 'texture_enhance' in i:
            net.texture_enhance.requires_grad_(False)
        elif 'fcs' in i:
            net.projection_local.requires_grad_(False)
            net.project_final.requires_grad_(False)
            net.ensemble_classifier_fc.requires_grad_(False)
        else:
            if 'xception' in str(type(net.net)):
                for j in net.net.seq:
                    if j[0]==i:
                        for t in j[1]:
                            t.requires_grad_(False)
            
            if 'EfficientNet' in str(type(net.net)):
                if i=='b0':
                    net.net._conv_stem.requires_grad_(False)
                stage_map=net.net.stage_map
                for c in range(len(stage_map)-2,-1,-1):
                    if not stage_map[c]:
                        stage_map[c]=stage_map[c+1]
                for c1,c2 in zip(stage_map,net.net._blocks):
                    if c1==i:
                        c2.requires_grad_(False)

    net=nn.SyncBatchNorm.convert_sync_batchnorm(net).to(local_rank)
    net = nn.parallel.DistributedDataParallel(net,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    AG=AGDA(**config.AGDA_config).to(local_rank)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, betas=config.adam_betas, weight_decay=config.weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)
    if config.ckpt:
        loc = 'cuda:{}'.format(local_rank)
        checkpoint = torch.load(config.ckpt, map_location=loc)
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch'])+1
        if load_state(net.module,checkpoint['state_dict']) and config.resume_optim:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            except:
                pass
        else:
            net.module.auxiliary_loss.alpha=torch.tensor(config.alpha)
        del checkpoint
    torch.cuda.empty_cache()
    for epoch in range(start_epoch, config.epochs):
        logs['epoch'] = epoch
        train_sampler.set_epoch(epoch)
        train_sampler.dataset.next_epoch()
        run(logs=logs,data_loader=train_loader,net=net,optimizer=optimizer,local_rank=local_rank,config=config,AG=AG,phase='train')
        run(logs=logs,data_loader=validate_loader,net=net,optimizer=optimizer,local_rank=local_rank,config=config,phase='valid')
        net.module.auxiliary_loss.alpha*=config.alpha_decay
        scheduler.step()
        if local_rank==0:
            torch.save({
                    'logs': logs,
                    'state_dict': net.module.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state':scheduler.state_dict()}, 'checkpoints/'+config.name+'/ckpt_%s.pth'%epoch)
        dist.barrier()

def train_loss(loss_pack,config):
    if 'loss' in loss_pack:
        return loss_pack['loss']
    loss=config.ensemble_loss_weight*loss_pack['ensemble_loss']+config.aux_loss_weight*loss_pack['aux_loss']
    if config.AGDA_loss_weight!=0:
        loss+=config.AGDA_loss_weight*loss_pack['AGDA_ensemble_loss']+config.match_loss_weight*loss_pack['match_loss']
    return loss
    
def run(logs,data_loader,net,optimizer,local_rank,config,AG=None,phase='train'):
    if local_rank==0:
        print('start ',phase)
    if config.AGDA_loss_weight==0:
        AG=None
    recorder={}
    if config.feature_layer=='logits':
        record_list=['loss','acc']
    else:
        record_list=['ensemble_loss','aux_loss','ensemble_acc']
        if AG is not None:
            record_list+=['AGDA_ensemble_loss','match_loss']
    for i in record_list:
        recorder[i]=dist_average(local_rank)
    # begin training
    start_time = time.time()
    if phase=='train':
        net.train()
    else: net.eval()
    for i, (X, y) in enumerate(data_loader):
        X = X.to(local_rank,non_blocking=True)
        y = y.to(local_rank,non_blocking=True)
        with torch.set_grad_enabled(phase=='train'):
            loss_pack=net(X,y,train_batch=True,AG=AG)
        if phase=='train':
            batch_loss = train_loss(loss_pack,config)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            if config.feature_layer=='logits':
                loss_pack['acc']=ACC(loss_pack['logits'],y)
            else:
                loss_pack['ensemble_acc']=ACC(loss_pack['ensemble_logit'],y)
        for i in record_list:
            recorder[i].step(loss_pack[i])

    # end of this epoch
    batch_info=[]
    for i in record_list:
        mesg=recorder[i].get()
        logs[i]=mesg
        batch_info.append('{}:{:.4f}'.format(i,mesg))
    end_time = time.time()

    # write log for this epoch
    if local_rank==0:
        logging.info('{}: {}, Time {:3.2f}'.format(phase,'  '.join(batch_info), end_time - start_time))


def distributed_train(config,world_size=0,num_gpus=0,rank_offset=0):
    if not num_gpus:
        num_gpus = torch.cuda.device_count()
    if not world_size:
        world_size=num_gpus
    mp.spawn(main_worker, nprocs=num_gpus, args=(world_size,rank_offset,config))
    torch.cuda.empty_cache()
