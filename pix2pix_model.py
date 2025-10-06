#this code is based on the pix2pix implementation by Jun-Yan Zhu : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

#this software is provided as is, and is not guaranteed to work or be suitable for any particular purpose
#copyright 2023-2024 Panagiotis Michalatos : pan.michalatos@gmail.com
#_______________________________________________________________________________________________________________________

from typing import Union
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from torchvision import transforms
import os
import json

import PIL.Image as PIL

import numpy as np
import cv2

from common_loss import *
from common_model import *


class Pix2PixModelInfo():
    def __init__(self, channels_A:int = 3, channels_B:int = 3, image_size: int = 64, number_of_downsamples:int = 6):
                
        self.channels_A = channels_A
        self.channels_B = channels_B
        
        self.image_size = image_size
        self.norm_layer_type = NORMALIZATION_LAYER.BATCHNORM

        #generator parameters and model
        self.ngf = 64 # number of gen filters in the last conv layer
        self.number_of_downsamples = number_of_downsamples

        #discriminator parameters and model
        self.ndf = 64 # number of discrim filters in the first conv layer
        self.disciminator_layers = 3

        #optimizer parameters
        self.learning_rate = 0.0002
        self.beta1 = 0.5
        self.lambda_L1 = 100.0 #weight for L1 loss

        #loss parameters
        self.GANLossMode = GANLOSS.VANILLA
        self.latest_losses = {}
        self.epochs_trained = 0

    @property
    def norm_layer(self):
        return nn.BatchNorm2d if self.norm_layer_type == NORMALIZATION_LAYER.BATCHNORM else nn.InstanceNorm2d

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    
    def save(self, file:str):
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(self, f, default=lambda o: o.__dict__, sort_keys=False, indent=4)

    @staticmethod
    def fromDict(d:dict):
        info =  Pix2PixModelInfo()
        info.__dict__.update(d)
        return info
    
    @staticmethod
    def fromJSON(json_str:str):
        j = json.loads(json_str)
        return Pix2PixModelInfo.fromDict(j)
    
    @staticmethod
    def load(file:str):
        with open(file, 'r', encoding='utf-8') as f:
            j = json.load(f)
            return Pix2PixModelInfo.fromDict(j)

class Pix2PixModel():
    G_WEIGHTS_FILE = 'generator_weights.pth'
    D_WEIGHTS_FILE = 'discriminator_weights.pth'
    JSON_MODEL_INFO_FILE = 'model_info.json'

    @staticmethod 
    def __ensureModelFileExists(model_folder:str, file:str) -> str:
        file_path = os.path.join(model_folder, file)
        if not os.path.exists(file_path):
            raise Exception(f'file {file_path} does not exist')
        return file_path

    @staticmethod
    def loadModelInfo(model_folder:str)->Pix2PixModelInfo:
        model_info_file = Pix2PixModel.__ensureModelFileExists(model_folder, Pix2PixModel.JSON_MODEL_INFO_FILE)        
        return Pix2PixModelInfo.load(model_info_file)

    @staticmethod
    def createFromFolder(model_folder:str, device:torch.device)->'Pix2PixModel':
        if not os.path.exists(model_folder):
            raise Exception(f'folder {model_folder} does not exist')
        
        inf = Pix2PixModel.loadModelInfo(model_folder)
        model = Pix2PixModel(inf.channels_A, inf.channels_B, inf.image_size, inf.number_of_downsamples, device, inf)
        model.__loadModelWeights(model_folder)      
   
        return model

    def __init__(self, channels_A : int, channels_B : int, image_size:int,  number_of_downsamples:int, device : torch.device, info:Pix2PixModelInfo = None):    
        self.device = device
        self.info = info or Pix2PixModelInfo(channels_A, channels_B, image_size, number_of_downsamples)   
        self.buildNetworks()    

    @property
    def current_learning_rate(self) -> float:
        return self.optimizer_G.param_groups[0]['lr']
    
    @property
    def current_epoch(self) -> int:
        return self.info.epochs_trained
    
    @property
    def current_losses(self):
        return self.info.latest_losses
    
    def buildNetworks(self):
        self.netG : nn.Module = UnetGenerator(
            self.info.channels_A, 
            self.info.channels_B, 
            self.info.number_of_downsamples, 
            self.info.ngf, 
            norm_layer=self.info.norm_layer
            )        
        init_weights(self.netG)
        self.netG.to(self.device)

        self.netD : nn.Module = NLayerDiscriminator(
            self.info.channels_A + self.info.channels_B, 
            self.info.ndf, 
            n_layers=self.info.disciminator_layers, 
            norm_layer=self.info.norm_layer
            )        
        init_weights(self.netD)
        self.netD.to(self.device)
    
    def train(self) :
        # define loss functions
        self.criterionGAN = GANLoss(self.info.GANLossMode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.info.learning_rate, betas=(self.info.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.info.learning_rate, betas=(self.info.beta1, 0.999)) 
  
    def eval(self):
        self.netG.eval()
        self.netD.eval()

    def init_schedulers(self, normal_epochs:int, decay_epochs:int):
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - normal_epochs) / float(decay_epochs + 1)
            return lr_l

        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_rule)
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lambda_rule)

    def on_end_of_epoch(self):
        self.scheduler_G.step()
        self.scheduler_D.step()    
        self.info.epochs_trained += 1

    def save(self, folder : str):
        if not os.path.exists(folder):
            os.makedirs(folder)

        model_info_file = os.path.join(folder, self.JSON_MODEL_INFO_FILE)
        ensureBakFile(model_info_file)
        self.info.save(model_info_file)

        self.__saveModelWeights(folder)

    def __saveModelWeights(self, folder : str):
        if not os.path.exists(folder):
            os.makedirs(folder)

        g_weights_files = os.path.join(folder, self.G_WEIGHTS_FILE)
        d_weights_files = os.path.join(folder, self.D_WEIGHTS_FILE)

        ensureBakFile(g_weights_files)
        ensureBakFile(d_weights_files)

        torch.save(self.netG.state_dict(), g_weights_files)
        torch.save(self.netD.state_dict(), d_weights_files)

    def __loadModelWeights(self, folder : str):
        if not os.path.exists(folder):
            raise Exception(f'folder {folder} does not exist')
        
        g_weights_files = Pix2PixModel.__ensureModelFileExists(folder, self.G_WEIGHTS_FILE)
        d_weights_files = Pix2PixModel.__ensureModelFileExists(folder, self.D_WEIGHTS_FILE)

        self.netG.load_state_dict(torch.load(g_weights_files, map_location=self.device))
        self.netD.load_state_dict(torch.load(d_weights_files, map_location=self.device))

    def __set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def __backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB : torch.Tensor = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake : torch.Tensor = self.netD(fake_AB.detach())
        self.loss_D_fake : torch.Tensor = self.criterionGAN(pred_fake, False)
        # Real
        real_AB : torch.Tensor = torch.cat((self.real_A, self.real_B), 1)
        pred_real : torch.Tensor = self.netD(real_AB)
        self.loss_D_real : torch.Tensor = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D : torch.Tensor = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def __backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB : torch.Tensor = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake : torch.Tensor = self.netD(fake_AB)
        self.loss_G_GAN : torch.Tensor = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 : torch.Tensor = self.criterionL1(self.fake_B, self.real_B) * self.info.lambda_L1
        # combine loss and calculate gradients
        self.loss_G : torch.Tensor = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        #self.netD.requires_grad_(True)  # enable backprop for D
        self.__set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.__backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.__set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.__backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

        
        self.info.latest_losses['loss_G'] = self.loss_G.item()
        self.info.latest_losses['loss_D'] = self.loss_D.item()

    def applyToImage(self, image : Union[torch.Tensor, np.ndarray], resize_and_crop : bool = True, normalize_input : bool = True) -> torch.Tensor:
        image_type = 'unknown'
        if isinstance(image, np.ndarray):
            image_type = 'cv2'
            #convert cv2 image to tensor
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else :
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        elif isinstance(image, torch.Tensor):
            image_type = 'tensor'
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
        elif isinstance(image, PIL.Image):
            image_type = 'pil'
            image = VF.to_tensor(image).unsqueeze(0)
        else:
            raise Exception(f'unsupported image type {type(image)}')

        if self.info.channels_A == 1 and image.shape[1] == 3:
            image = VF.rgb_to_grayscale(image)
        elif self.info.channels_A == 3 and image.shape[1] == 1:
            image = torch.cat([image]*3, 1)

        if resize_and_crop:
            image = VF.resize(image, self.info.image_size)
            image = VF.center_crop(image, self.info.image_size)

        if normalize_input:
            image = VF.normalize(image, [0.5]*self.info.channels_A, [0.5]*self.info.channels_A)

        image = image.to(self.device)
        generated_image = self.netG(image)

        #denormalize
        generated_image = VF.normalize(generated_image, [-1.0]*self.info.channels_B, [2.0]*self.info.channels_B)

        if image_type == 'cv2':
            generated_image = torch.clamp(generated_image, 0.0, 1.0)
            generated_image = generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy()            
            generated_image = (generated_image * 255).astype(np.uint8)
            generated_image = cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)
        elif image_type == 'tensor':
            generated_image = generated_image.squeeze(0)
        elif image_type == 'pil':
            generated_image = torch.clamp(generated_image, 0.0, 1.0)
            generated_image = VF.to_pil_image(generated_image.squeeze(0))

        return generated_image