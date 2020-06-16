"""
This file loads the padded data of CIFAR10.
"""
import torchvision
from torchvision import datasets, transforms
import torch

def load_pading_training_data(device):
  transform_padding = transforms.Compose([
    transforms.Pad(padding=4),
    transforms.ToTensor(),
  ])
  trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_padding)
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=True)

  for batch_idx, (data, target) in enumerate(train_loader):
        cifar_images, cifar_labels = data.to(device), target.to(device)
  cifar_images[:,:,:4,:] = 0.5
  cifar_images[:,:,-4:,:] = 0.5
  cifar_images[:,:,:,:4] = 0.5
  cifar_images[:,:,:,-4:] = 0.5

  return cifar_images, cifar_labels