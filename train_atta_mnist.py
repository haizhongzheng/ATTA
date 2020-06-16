"""
This file trains ATTA models on MNIST dataset.
"""
from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
import json

from models.small_cnn import *

import adv_attack

parser = argparse.ArgumentParser(description='PyTorch MNIST TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.3, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', type=int, default=40,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.01, type=float,
                    help='perturb step size')
parser.add_argument('--beta', type=int, default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='data-model/test',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--gpuid', default='0',
                    help='gpuid')
parser.add_argument('--training-method', default='mat',
                    help='Adversarial training method: mat or trades')
parser.add_argument('--epochs-reset', type=int, default=100, metavar='N',
                    help='number of epochs to reset perturbation')

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
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=60000, shuffle=True)

def build_dataset(train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        mnist_images, mnist_labels = data.to(device), target.to(device)
    return mnist_images, mnist_labels

def train(args, model, device, mnist_nat_x, mnist_x, mnist_y, optimizer, epoch):
    model.train()

    batch_size = args.batch_size
    num_of_example = 60000
    cur_order = np.random.permutation(num_of_example)
    iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)
    batch_idx = -batch_size

    for i in range(iter_num):
        batch_idx = (batch_idx + batch_size) if batch_idx + batch_size < num_of_example else 0
        x_batch = mnist_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]
        x_nat_batch = mnist_nat_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]
        y_batch = mnist_y[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]

        optimizer.zero_grad()

        # calculate robust loss
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
          loss = criterion_ce(F.log_softmax(model(x_adv_next), dim=1), y_batch)
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
        mnist_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]] = x_adv_next

        # print progress
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, num_of_example,
                       100. * batch_idx / num_of_example, loss.item()))
            print(torch.min(x_adv_next - x_nat_batch))
            print(torch.max(x_adv_next - x_nat_batch))

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 55:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # init model, Net() can be also used here for training
    model = SmallCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    mnist_x, mnist_y = build_dataset(train_loader)
    mnist_nat_x = mnist_x.clone()
    mnist_x = mnist_x + 0.001 * torch.randn(mnist_x.shape).cuda().detach()
    print(mnist_x.shape)
    print(mnist_y.shape)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, mnist_nat_x, mnist_x, mnist_y, optimizer, epoch)

        print('================================================================')

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-mnist-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-mnist-epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
