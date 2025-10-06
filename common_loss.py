#this software is provided as is, and is not guaranteed to work or be suitable for any particular purpose
#copyright 2023-2024 Panagiotis Michalatos : pan.michalatos@gmail.com
#this code is based on the pix2pix implementation by Jun-Yan Zhu : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
#_______________________________________________________________________________________________________________________

import torch
import torch.nn as nn
from enum import Enum

class GANLOSS(str, Enum):
    VANILLA = 'vanilla'
    LSGAN = 'lsgan'
    WGAN_GP = 'wgan_gp'

class GANLoss(nn.Module):
    def __init__(self, gan_mode : GANLOSS = GANLOSS.VANILLA, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == GANLOSS.LSGAN:
            self.loss = nn.MSELoss()
        elif gan_mode == GANLOSS.VANILLA:
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == GANLOSS.WGAN_GP:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor : torch.Tensor = self.real_label
        else:
            target_tensor : torch.Tensor  = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in [GANLOSS.LSGAN, GANLOSS.VANILLA]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == GANLOSS.WGAN_GP:
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss