## ATTA (Efficient *A*dversarial *T*raining with *T*ransferable *A*dversarial Examples)

Code for [CVPR'2020](https://arxiv.org/abs/1912.11969) paper: *Efficient Adversarial Training with Transferable Adversarial Examples*.

## Prerequisites
- Python 3.6.3
- Pytorch 1.2.0, torchvision 0.4.0
- Numpy 1.13.3

## Code Overview
The directory `models` contains model architecture definition files.
The directory `data-config` contains different config files to train the model and the directory `data-model` is used to contain model checkpoints.

Other seven Python scripts are used to train and evaluate the ATTA model.

- `train_atta_mnist.py`: trains ATTA models on MNIST dataset.
- `train_atta_cifar.py`: trains ATTA models on CIFAR10 dataset.
- `cifar_dataloader.py`: loads the padded data of CIFAR10.
- `adv_attack.py`: generates accumulative adversarial examples for the training of ATTA.
- `adaptive_data_aug.py`: performs data augmentation and inverse data augmentation for ATTA.
- `pgd_attack_mnist.py`: performs PGD-k attack on MNIST models.
- `pgd_attack_cifar10.py`: performs PGD-k attack on CIFAR10 models.

## Simple instructions to train and evaluate models

### Train a model:
```
#MNIST

python train_atta_mnist.py --config-file [config_file_name] --gpuid [GPU_ID]


#CIFAR10

python train_atta_cifar.py --config-file [config_file_name] --gpuid [GPU_ID]
```


### Attack a model:
```
#MNIST

python pgd_attack_mnist.py --model-dir [path of model] --gpuid [GPU_ID]


#CIFAR10

python pgd_attack_cifar10.py --model-dir [path of model] --gpuid [GPU_ID]
```

### Naming rule for configuration files in `data-config`:

[`dataset`]-atta-[`the number of attack iterations`]-[`training method`].json

`data-config/mnist-atta-1-mat.json` means that model will be trained with MAT(ATTA-1) on MNIST.

### Examples for training and evaluate:

- TRADES(ATTA-1) on MNIST:

```
python train_atta_mnist.py --config-file data-config/mnist-atta-1-trades.json --gpuid 0

python pgd_attack_mnist.py --model-dir data-model/mnist-trades-atta-1/model-mnist-epoch60.pt --gpuid 0
```

- MAT(ATTA-1) on CIFAR10:

```
python train_atta_cifar.py --config-file data-config/cifar-atta-1-mat.json --gpuid 0

python pgd_attack_cifar10.py --model-dir data-model/cifar-mat-atta-1/model-cifar-epoch38.pt --gpuid 0
```

## Reference

```
@inproceedings{zheng2020efficient,
 author={Zheng, Haizhong and Zhang, Ziqi and Gu, Juncheng and Lee, Honglak and Prakash, Atul},
 title={Efficient Adversarial Training with Transferable Adversarial Examples},
 BOOKTITLE = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 YEAR = {2020}
}
```

Credit: The implementation of ATTA is based on [TRADES](https://github.com/yaodongyu/TRADES) code.