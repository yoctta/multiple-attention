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
from models.MAT import netrunc
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
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")
    dist.init_process_group(backend='nccl', init_method='file://%s/.pytorch_distribute'%os.getcwd(),world_size=world_size, rank=rank)
    if rank==0:
        try:
            os.remove('.pytorch_distribute')
        except:
            pass
    numpy.random.seed(1234567)
    torch.manual_seed(1234567)
    torch.cuda.manual_seed(1234567)
    torch.cuda.set_device(local_rank)
    train_dataset = DeepfakeDataset(phase='train',**config.train_dataset)
    validate_dataset=DeepfakeDataset(phase='val',**config.val_dataset)
    train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
    validate_sampler=torch.utils.data.distributed.DistributedSampler(validate_dataset)
    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,sampler=train_sampler,pin_memory=True,num_workers=config.workers)
    validate_loader=torch.utils.data.DataLoader(validate_dataset, batch_size=config.batch_size,sampler=validate_sampler,pin_memory=True,num_workers=config.workers)
    logs = {}
    start_epoch = 0
    net = netrunc(config.net,config.feature_layer,config.num_classes,config.dropout_rate,config.pretrained)   
    net=nn.SyncBatchNorm.convert_sync_batchnorm(net).to(local_rank)
    net = nn.parallel.DistributedDataParallel(net,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
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

        
    for epoch in range(start_epoch, config.epochs):
        logs['epoch'] = epoch
        train_sampler.set_epoch(epoch)
        train_sampler.dataset.next_epoch()
        run(logs=logs,data_loader=train_loader,net=net,optimizer=optimizer,local_rank=local_rank,config=config,phase='train')
        run(logs=logs,data_loader=validate_loader,net=net,optimizer=optimizer,local_rank=local_rank,config=config,phase='valid')
        scheduler.step()
        if local_rank==0:
            torch.save({
                    'logs': logs,
                    'state_dict': net.module.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state':scheduler.state_dict()}, 'checkpoints/'+config.name+'/ckpt_%s.pth'%epoch)
        dist.barrier()

    
def run(logs,data_loader,net,optimizer,local_rank,config,phase='train'):
    if local_rank==0:
        print('start ',phase)
    recorder={}
    record_list=['loss','acc']
    for i in record_list:
        recorder[i]=dist_average(local_rank)
    # begin training
    start_time = time.time()
    if phase=='train':
        net.train()
    else: net.eval()
    for i, (X, y) in enumerate(data_loader):
        loss_pack={}
        X = X.to(local_rank,non_blocking=True)
        y = y.to(local_rank,non_blocking=True)
        with torch.set_grad_enabled(phase=='train'):
            logit=net(X)
            batch_loss = F.cross_entropy(logit,y)
        if phase=='train':
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            loss_pack['loss']=batch_loss
            loss_pack['acc']=ACC(logit,y)
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

if __name__=="__main__":
    from config import train_config
    for feature in range(2,5):
        feature_layer='b%s'%feature
        name='EFB4_ALL_c23_trunc_%s'%feature_layer
        Config=train_config(name,['ff-all-c23','efficientnet-b4'],attention_layer='',feature_layer=feature_layer,epochs=20,batch_size=10,augment='augment1')
        Config.mkdirs()
        distributed_train(Config) 
