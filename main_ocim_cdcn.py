import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from configs import get_args
from augmentations import get_aug
from models import get_model
#from tools import AverageMeter, PlotLogger
from datasets import get_dataset_liveness
from optimizers import get_optimizer, LR_Scheduler

from utils.eval_ocim import *

import time

#torch.backends.cudnn.enabled = False 
#torch.backends.cudnn.benchmark = False

def main(args):
    if args.cudnn:
        pass
    else:
        torch.backends.cudnn.enabled = False 
        torch.backends.cudnn.benchmark = False

    train_set = get_dataset_liveness(
        args.dataset, 
        args.data_dir, 
        transform=get_aug(args.model, args.image_size, True), 
        train=True, 
        download=args.download, # default is False
        debug_subset_size=args.batch_size if args.debug else None # run one batch if debug
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_set = get_dataset_liveness(
        args.dataset, 
        args.test_dir, 
        transform=get_aug(args.model, args.image_size, False), 
        train=False, 
        download=args.download, # default is False
        debug_subset_size=args.batch_size if args.debug else None # run one batch if debug
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # define model
    model = get_model(args.model, args.backbone).to(args.device)
    if 'simsiam' in args.model and args.proj_layers is not None: model.projector.set_layers(args.proj_layers)
    model = torch.nn.DataParallel(model)
    #if torch.cuda.device_count() > 1: model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # define optimizer
    optimizer = get_optimizer(
        args.optimizer, model, 
        lr=args.base_lr*args.batch_size/256, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.warmup_epochs, args.warmup_lr*args.batch_size/256, 
        args.num_epochs, args.base_lr*args.batch_size/256, args.final_lr*args.batch_size/256, 
        len(train_loader)
    )

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_us = AverageMeter('Loss-us', ':.6f')
    losses_ce = AverageMeter('Loss-ce', ':.6f')
    losses_tt = AverageMeter('Loss-tt', ':.6f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    learning_rate = AverageMeter('LR', ':.6f')
    # Start training
    #global_progress = tqdm(range(0, args.stop_at_epoch), desc=f'Training')
    for epoch in range(args.stop_at_epoch):
        batch_time.reset()
        data_time.reset()
        losses_us.reset()
        losses_ce.reset()
        losses_tt.reset()
        top1.reset()
        learning_rate.reset()
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses_us, losses_ce, losses_tt, top1, learning_rate],
            prefix="Epoch: [{}]".format(epoch))

        model.train()

        end = time.time()

        for idx, ((images1, images2), labels, true_labels) in enumerate(train_loader):
            labels = labels.to(args.device)
            true_labels = true_labels.to(args.device)
            images1 = images1.to(args.device)
            images2 = images2.to(args.device)

            model.zero_grad()

            losses, (c1, c2) = model.forward(images1, images2, labels, true_labels)

            for key in losses.keys():
                losses[key] = losses[key].double().sum() / float(torch.cuda.device_count())

            if epoch < args.warmup_epochs:
                #losses['loss_us'].backward()
                losses['loss_tt'].backward()
            else:
                losses['loss_tt'].backward()
            optimizer.step()
            #loss_meter.update(loss.item())

            losses_us.update(losses['loss_us'].item())
            losses_ce.update(losses['loss_ce'].item())
            losses_tt.update(losses['loss_tt'].item())

            acc1 = accuracy(c1, labels, (1, ))[0]
            acc2 = accuracy(c2, labels, (1, ))[0]

            top1.update((acc1.item() + acc2.item()) / 2.)

            batch_time.update(time.time() - end)
            end = time.time()


            lr = lr_scheduler.step()

            learning_rate.reset()
            learning_rate.update(lr)

            if idx % args.print_freq == 0:
                progress.display(idx)
    
        '''eval'''
        print('Start evaluation:')
        model.eval()
        lines = []
        for idx, (images1, labels, true_labels, names) in enumerate(test_loader):
            #labels = labels
            #true_labels = true_labels
            images1 = images1.to(args.device)
    
            #images2 = images2.to(args.device)

            model.zero_grad()
            
            c1 = model.forward(images1)

            labels = labels.detach().cpu().numpy()
            c1 = c1.detach().cpu().numpy()

            len_bs = len(names)
            for l in range(len_bs):
                logits = c1[l]
                label = labels[l]
                true_label = true_labels[l]
                name = names[l]
                #print(idx * len_bs + l, name, logits, label)
                lines.append([name, logits, label, true_label])
        
        score_file = os.path.join(args.output_dir, 'score_file.txt')
        if not os.path.exists(score_file):
            performances = []
        else:
            with open(score_file, 'r') as fid:
                performances = fid.readlines()
        print('Dev-Test Inference Done!')
        performance = eval_ocim(lines)
        print(performance)
        performances.append(str(epoch) + ': ' + performance + '\n')
        with open(score_file, 'w') as fid:
            fid.writelines(performances)
        print('Scores saved to %s'%(score_file))
        # Save checkpoint
        
        model_path = os.path.join(args.output_dir, f'{args.model}-{args.dataset}-epoch{epoch+1}.pth')
        torch.save({
            'epoch': epoch+1,
            'state_dict':model.module.state_dict(),
            # 'optimizer':optimizer.state_dict(), # will double the checkpoint file size
            'lr_scheduler':lr_scheduler,
            'args':args,
        }, model_path)
        print(f"Model saved to {model_path}")
        

# classes
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if isinstance(output, tuple) or isinstance(output, list):
        output = output[0]

    if True:
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

if __name__ == "__main__":
    main(args=get_args())
















