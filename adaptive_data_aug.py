"""
This file performs data augmentation and inverse data augmentation for ATTA.
"""
from __future__ import print_function
import os
import sys
import torch
import numpy as np
import random


#Apply random crop and flip to the input
#Input: 3-dim tensor [batchsize, 3, 40, 40]
#Output: 3-dim tensor [batchsize, 3, 32, 32], transform information {{x,y}, flip}
def atta_aug(input_tensor, rst):
  batch_size = input_tensor.shape[0]
  x = torch.zeros(batch_size)
  y = torch.zeros(batch_size)
  flip = [False] * batch_size

  for i in range(batch_size):
    flip_t = bool(random.getrandbits(1))
    x_t = random.randint(0,8)
    y_t = random.randint(0,8)

    rst[i,:,:,:] = input_tensor[i,:,x_t:x_t+32,y_t:y_t+32]
    if flip_t:
      rst[i] = torch.flip(rst[i], [2])
    flip[i] = flip_t
    x[i] = x_t
    y[i] = y_t

  return rst, {"crop":{'x':x, 'y':y}, "flipped":flip}

def atta_aug_trans(input_tensor, transform_info, rst):
  batch_size = input_tensor.shape[0]
  x = transform_info['crop']['x']
  y = transform_info['crop']['y']
  flip = transform_info['flipped']
  for i in range(batch_size):
    flip_t = int(flip[i])
    x_t = int(x[i])
    y_t = int(y[i])
    rst[i,:,:,:] = input_tensor[i,:,x_t:x_t+32,y_t:y_t+32]
    if flip_t:
      rst[i] = torch.flip(rst[i], [2])
  return rst

#Apply random crop and flip to the input
#Input: 3-dim tensor [batchsize, 3, 40, 40], 3-dim tensor [batchsize, 3, 32, 32], transform information {{x,y}, flip}
#Output: 3-dim tensor [batchsize, 3, 40, 40]
def inverse_atta_aug(source_tensor, adv_tensor, transform_info):
  x = transform_info['crop']['x']
  y = transform_info['crop']['y']
  flipped = transform_info['flipped']
  batch_size = source_tensor.shape[0]

  for i in range(batch_size):
    flip_t = int(flipped[i])
    x_t = int(x[i])
    y_t = int(y[i])
    if flip_t:
      adv_tensor[i] = torch.flip(adv_tensor[i], [2])
    source_tensor[i,:,x_t:x_t+32,y_t:y_t+32] = adv_tensor[i]

  return source_tensor
