"""
This file generates accumulative adversarial examples for the training of ATTA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_adv_atta(model, x_natural, x_adv, y, step_size=0.003, epsilon=0.031,
num_steps=10, loss_type='mat'):
    # define KL-loss
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    if loss_type == 'mat':
        for i in range(num_steps):
            x_adv.requires_grad_()
            ce_loss = nn.CrossEntropyLoss()
            with torch.enable_grad():
                loss_kl = (1/batch_size) * ce_loss(F.log_softmax(model(x_adv), dim=1), y)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif loss_type == 'trades':
        nat_softmax = F.softmax(model(x_natural), dim=1)
        for i in range(num_steps):
            x_adv.requires_grad_()
            kl_div_loss = nn.KLDivLoss(size_average=False)
            with torch.enable_grad():
                loss_kl = kl_div_loss(F.log_softmax(model(x_adv), dim=1),
                                       nat_softmax)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        print("Unknown loss method.")
        raise

    return x_adv
