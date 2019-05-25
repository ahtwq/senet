import argparse
import os
import sys
import time
import torch
from torch import nn
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import utils
import models
import dataLoader
import tabulate
from PIL import Image, ImageFile
import numpy as np
import random

## args
parser = argparse.ArgumentParser(description='clarity training')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--se', type=bool, help='whether use se_net(default: No)')
parser.add_argument('--train_batchSize', type=int, default=32, metavar='N', help='input batch size (default: 32)')
parser.add_argument('--test_batchSize', type=int, default=10, metavar='N', help='input batch size (default: 10)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--lr_init', type=float, default=0.01, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=100, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
print(args)

## python train.py --dir=weight --epochs=25 --batch_size=30 --lr_init=0.0001 

## seed fix
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

print('Preparing directory {}'.format(args.dir))
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

## dataset
print('Loading dataset')
loaders = dataLoader.load_cifar10(args.train_batchSize, args.test_batchSize)
       
## model
print('Building model')
num_classes = 6
model = models.resnet20()
if args.se:
    model = models.se_resnet20()
    
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('Let us use {} GPUs'.format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
    model.cuda()

## loss func
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd)

## lr schedule
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

## cosine learning rate delay
def adjust_lr(optimizer, epoch, epochs, M=1, alpha_zero=0.001):
    cos_inner = np.pi * (epoch % (epochs // M))
    cos_inner /= epochs // M
    cos_out = np.cos(cos_inner) + 1
    cur_lr =  float(alpha_zero / 2 * cos_out)
    return cur_lr

    
## train model
print('Starting train model')
columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']
best_acc_on_dev = 0.0 
for epoch in range(0, args.epochs):
    time_ep = time.time()
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    #lr = adjust_lr(optimizer, epoch, args.epochs, M=1, alpha_zero=args.lr_init)
    #utils.adjust_learning_rate(optimizer, lr)
    
    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, num_classes)
    test_res = utils.eval(loaders['test'], model, criterion, num_classes)
    
    ## save model
    if test_res['accuracy'] >= best_acc_on_dev:
        # best_acc_on_dev = test_res['accuracy']
        utils.save_checkpoint(
            args.dir,
            epoch,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
    ## write info.
    txt = open(os.path.join(args.dir, 'conf_matrix.txt'), 'a+')
    txt.write('epoch' + str(epoch) + '\n')
    txt.write('-'*20 + '\n')
    txt.write('va_loss:{:.6f}, va_acc:{:.6f}'.format(test_res['loss'], test_res['accuracy']) + '\n')
    txt.write(str(test_res['conf_matrix']) + '\n')
    txt.write('\n')
    txt.close()

    time_ep = (time.time() - time_ep) / 60
    values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'], time_ep]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.6f')
    if epoch % 10 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
    
print('training is over, please test soon.')
