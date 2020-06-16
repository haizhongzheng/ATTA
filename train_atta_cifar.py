"""
This file trains ATTA models on CIFAR10 dataset.
"""
from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable

from models.wideresnet import *
from adaptive_data_aug import atta_aug, atta_aug_trans, inverse_atta_aug
import json

import numpy as np

import cifar_dataloader
import adv_attack

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=38, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--epochs-reset', type=int, default=10, metavar='N',
                    help='number of epochs to reset perturbation')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007, type=float,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--gpuid', type=int, default=0,
                    help='The ID of GPU.')
parser.add_argument('--training-method', default='mat',
                    help='Adversarial training method: mat or trades')

parser.add_argument('--config-file',
                    help='The path of config file.')


args = parser.parse_args()

if (args.config_file is None):
    pass
else:
    with open(args.config_file) as config_file:
        config = json.load(config_file)
        args.model_dir = config['model-dir']
        args.num_steps = config['num-steps']
        args.step_size = config['step-size']
        args.epochs_reset = config['epochs-reset']
        args.epsilon = config['epsilon']
        args.beta = config['beta']
        args.training_method = config['training-method']


epochs_reset = args.epochs_reset
training_method = args.training_method
beta = args.beta

#Config file will overlap commend line args

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

# setup data loader

def train(args, model, device, cifar_nat_x, cifar_x, cifar_y, optimizer, epoch):
    model.train()
    num_of_example = 50000
    batch_size = args.batch_size
    cur_order = np.random.permutation(num_of_example)
    iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)
    batch_idx = -batch_size

    print(batch_size)
    for i in range(iter_num):
        batch_idx = (batch_idx + batch_size) if batch_idx + batch_size < num_of_example else 0
        x_batch = cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]
        x_nat_batch = cifar_nat_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]
        y_batch = cifar_y[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]

        batch_size = y_batch.shape[0]

        #atta-aug
        rst = torch.zeros(batch_size,3,32,32).to(device)
        x_batch, transform_info = atta_aug(x_batch, rst)
        rst = torch.zeros(batch_size,3,32,32).to(device)
        x_nat_batch = atta_aug_trans(x_nat_batch, transform_info, rst)

        x_adv_next = adv_attack.get_adv_atta(
                           model=model,
                           x_natural=x_nat_batch,
                           x_adv=x_batch,
                           y=y_batch,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           num_steps=args.num_steps,
                           loss_type="mat"
        )

        model.train()
        x_adv_next = Variable(x_adv_next, requires_grad=False)
        optimizer.zero_grad()

        if training_method == "mat":
          criterion_ce = nn.CrossEntropyLoss()
          loss = (1.0 / batch_size) * criterion_ce(F.log_softmax(model(x_adv_next), dim=1), y_batch)
        elif training_method == "trades":
          criterion_kl = nn.KLDivLoss(size_average=False)
          nat_logits = model(x_nat_batch)
          loss_natural = F.cross_entropy(nat_logits, y_batch)
          loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv_next), dim=1),F.softmax(nat_logits, dim=1))
          loss = loss_natural + beta * loss_robust
        else:
          print("Unknown loss method.")
          raise

        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, num_of_example,
                       100. * batch_idx / num_of_example, loss.item()))
            print(torch.min(x_adv_next - x_nat_batch))
            print(torch.max(x_adv_next - x_nat_batch))

        cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]] = inverse_atta_aug(
            cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]],
            x_adv_next, transform_info)

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 30:
        lr = args.lr * 0.1
    if epoch >= 36:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    model = WideResNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cifar_x, cifar_y = cifar_dataloader.load_pading_training_data(device)
    cifar_nat_x = cifar_x.clone()
    cifar_x = cifar_x.detach() + 0.001 * torch.randn(cifar_x.shape).cuda().detach()

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        #reset perturbation
        if epoch % epochs_reset == 0:
            cifar_x = cifar_nat_x.clone()
            cifar_x = cifar_x.detach() + 0.001 * torch.randn(cifar_x.shape).cuda().detach()
        train(args, model, device, cifar_nat_x, cifar_x, cifar_y, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-cifar-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-cifar-epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()