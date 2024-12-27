import argparse
import os
import random
import time
import numpy as np
import torch
from nets.swin_transformer_v2 import swin_v2_b, load_pretrain
from dataloder import get_train_dataset, get_val_dataset
from torch.utils.data import DataLoader
from utils import AverageMeter, ProgressMeter, accuracy
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

class Config:
    train_root = 'path/to/train/dataset'
    val_root = 'path/to/val/dataset'
    train_list = 'path/to/train/dataset/list'
    val_list = 'path/to/val/dataset/list'
    batch_size = 64
    epochs = 100
    lr = 0.001
    weight_decay = 1e-4
    momentum = 0.9
    pretrain = 'path/to/pre-trained/model'
    save_dir = './checkpoints'
    print_freq = 10
    local_rank = 0

def main():
    args = Config()

    # Set up distributed training
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    cudnn.benchmark = True

    # Set random seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Create model
    model = swin_v2_b(num_classes=2, fp16=False)
    if args.pretrain:
        model = load_pretrain(model, pretrain=args.pretrain)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Load datasets
    train_dataset = get_train_dataset(args.train_root, args.train_list, 224, args)
    val_dataset = get_val_dataset(args.val_root, args.val_list, 224, args)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=val_sampler)

    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train(train_loader, model, criterion, optimizer, epoch, args)
        validate(val_loader, model, criterion, args)

        if (epoch + 1) % 10 == 0:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'swin_v2_epoch_{epoch+1}.pth'))

def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1], prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(val_loader, model, criterion, args):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1], prefix='Test: ')

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

if __name__ == '__main__':
    main()
