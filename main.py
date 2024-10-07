import os
import random
import numpy as np
import argparse
import time
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import datetime
import wandb
from dataset.dataset_7_2 import ClassBalancedDataset, Dataset
from timm.data import create_transform
from models.MTSD_CF import SwinTransFER01_01
parser = argparse.ArgumentParser()
from timm.scheduler.cosine_lr import CosineLRScheduler

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--dataset', type=str,default='RAF-DB')
parser.add_argument('--data-path', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--checkpoint_path', type=str, default='./checkpoin/' + time_str + 'model.pth')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/'+time_str+'model_best.pth')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=3.125e-5, type=float, metavar='LR', dest='lr')
parser.add_argument('--lr_min', default=5e-7, type=float)
parser.add_argument('--warmup_lr_init', default=5e-8, type=float)
parser.add_argument('--wd', '--weight-decay', default=0.05, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--beta', default=0.3, type=float)
parser.add_argument('--gama', default=0.9, type=float)
args = parser.parse_args()

if args.dataset=='RAF-DB':
    args.data = args.data_path
elif args.dataset=='SFEW':
    args.data = args.data_path
elif args.dataset=='FERPlus':
    args.data = args.data_path
elif args.dataset=='Oulu_CASIA':
    args.data= args.data_path


def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0
    print('Training time: ' + now.strftime("%m-%d %H:%M"))

    # create model
    model = SwinTransFER01_01(args.checkpoint)
    model = torch.nn.DataParallel(model).cuda()

    if args.dataset=='FERPlus':
        model.module.head_fg = nn.Linear(768, 8).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

    recorder = RecorderMeter(args.epochs)

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    if args.dataset=='RAF-DB':
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=60000,
            cycle_mul=1.,
            lr_min=args.lr_min,
            warmup_lr_init=args.warmup_lr_init,
            warmup_t=2000,  ### steps from warmup_lr to min_lr
            cycle_limit=1,
            t_in_epochs=False,
        )
        train_transforms= create_transform(
            input_size=112,
            scale=(0.8, 1.0),
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
    elif args.dataset=='Oulu_CASIA':
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=7200,
            cycle_mul=1.,
            lr_min=args.lr_min,
            warmup_lr_init=args.warmup_lr_init,
            warmup_t=360,  ### steps from warmup_lr to min_lr
            cycle_limit=1,
            t_in_epochs=False,
        )
        train_transforms= create_transform(
            input_size=112,
            scale=(0.8, 1.0),
            ratio=(3./4., 4./3.),
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
    elif args.dataset=='SFEW':
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=60000,
            cycle_mul=1.,
            lr_min=args.lr_min,
            warmup_lr_init=args.warmup_lr_init,
            warmup_t=1000,  ### steps from warmup_lr to min_lr
            cycle_limit=1,
            t_in_epochs=False,
        )
        train_transforms= create_transform(
            input_size=112,
            scale=(0.8, 1.0),
            ratio=(3./4., 4./3.),
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
        
    
    eval_transforms = transforms.Compose([
                transforms.Resize([112, 112]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    train_dataset = Dataset(traindir, transform=train_transforms)
    if args.dataset == 'FERPlus':
        train_dataset = ClassBalancedDataset(train_dataset)
    test_dataset = Dataset(valdir, transform=eval_transforms)


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    step=0
    for epoch in range(0, args.epochs):
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print('Current learning rate: ', current_learning_rate)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        # train for one epoch
        train_acc, train_los,step= train(train_loader, model, criterion, optimizer,scheduler ,epoch, step,args)

        # evaluate on validation set
        val_acc, val_los = validate(val_loader, model, criterion, args)

        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        curve_name = time_str + 'cnn.png'
        recorder.plot_curve(os.path.join('./log/', curve_name))

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print('Current best accuracy: ', best_acc.item())

        if args.wandb:
            wandb.log({
                "Train_Loss": train_los,
                "Test_Accuracy": val_acc,
            })
            config=wandb.config
            config.update({"best_test_acc":best_acc},allow_val_change=True)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

        save_checkpoint({'state_dict': model.state_dict()}, is_best, args)
        end_time = time.time()
        epoch_time = end_time - start_time
        print("An Epoch Time: ", epoch_time)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(str(epoch_time) + '\n')
    with open('./result.txt', 'a') as f:
        f.write(str(args.dataset)+'  alpha: '+str(args.alpha)+'  beta: '+str(args.beta)+'  gamma: '+str(args.gama)+'  Acc: '+str(best_acc.item()) + '\n')


def train(train_loader, model, criterion, optimizer, scheduler,epoch, step,args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (imgs, target_7,target_2,target_4) in enumerate(train_loader):

        imgs = imgs.cuda()
        target_7 = target_7.cuda()
        target_2 = target_2.cuda()
        target_4 = target_4.cuda()
        # compute output
        output_7,output_2,output_4,feature_fg,feature_fg_convert1,feature_fg_convert2 = model(imgs)
        feature_fg = F.log_softmax(feature_fg,dim=1)

        kl = nn.KLDivLoss(reduction='batchmean')     
        
        loss= (args.alpha)*criterion(output_7, target_7) + (1-args.alpha-args.beta)*criterion(output_2, target_2) + args.beta*criterion(output_4, target_4) + args.gama*kl(feature_fg,F.softmax(feature_fg_convert2,dim=1)) + (1-args.gama)*kl(feature_fg,F.softmax(feature_fg_convert1,dim=1))   

        acc1, _ = accuracy(output_7, target_7, topk=(1, 5))
        losses.update(loss.item(), imgs.size(0))
        top1.update(acc1[0], imgs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        step=step+1
        scheduler.step_update(step)
        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)
    return top1.avg, losses.avg,step


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (imgs, target_7,target_2,target_4) in enumerate(val_loader):
            imgs = imgs.cuda()
            target_7 = target_7.cuda()
            target_2=target_2.cuda()
            target_4=target_4.cuda()

            # compute output
            output_7,output_2,output_4,feature_fg,feature_fg_convert1,feature_fg_convert2= model(imgs)
            feature_fg = F.log_softmax(feature_fg,dim=1)

            kl = nn.KLDivLoss(reduction='batchmean')     
        
            loss= (args.alpha)*criterion(output_7, target_7) + (1-args.alpha-args.beta)*criterion(output_2, target_2) + args.beta*criterion(output_4, target_4) + args.gama*kl(feature_fg,F.softmax(feature_fg_convert2,dim=1)) + (1-args.gama)*kl(feature_fg,F.softmax(feature_fg_convert1,dim=1))  
            
            # measure accuracy and record loss
            acc, _ = accuracy(output_7, target_7, topk=(1, 5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(acc[0], imgs.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

            
        print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        with open('./log/' + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    if is_best:
        shutil.copyfile(args.checkpoint_path, args.best_checkpoint_path)


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
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


if __name__ == '__main__':
    main()