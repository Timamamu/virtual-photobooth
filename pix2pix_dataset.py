#this software is provided as is, and is not guaranteed to work or be suitable for any particular purpose
#copyright 2023-2024 Panagiotis Michalatos : pan.michalatos@gmail.com
#this code is based on the pix2pix implementation by Jun-Yan Zhu : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
#_______________________________________________________________________________________________________________________

import os
import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as VF
import torch.utils.data
import torch.nn as nn
from PIL import Image


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def isImageFile(filename : str):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def gatherImageFiles(folder : str):
    image_files = os.listdir(folder)
    image_files = [i for i in image_files if isImageFile(i)]
    return image_files

#assumes that images with same name exist in both folders A and B and are of the same size
class ABPairedDataset(torch.utils.data.Dataset): 
    def __init__(self, 
                 folder_A : str, folder_B : str, 
                 channels_A : int, channels_B : int, 
                 img_size : int,
                 pre_transform_A : list[nn.Module] = None, pre_transform_B : list[nn.Module] = None,
                 post_transform_A : list[nn.Module] = None, post_transform_B : list[nn.Module] = None
                 ):
        super().__init__()

        self.folder_A = folder_A
        self.folder_B = folder_B

        self.channels_A = channels_A
        self.channels_B = channels_B

        self.img_size = img_size

        images_A =  gatherImageFiles(folder_A)
        images_B =  gatherImageFiles(folder_B)

        #find intersection (filenames that exist in both folders)
        self.image_files = [i for i in images_A if i in images_B]

        self.pre_transform_A = transforms.Compose(pre_transform_A) if pre_transform_A is not None else None
        self.pre_transform_B = transforms.Compose(pre_transform_B) if pre_transform_B is not None else None

        self.post_transform_A = [
            transforms.ToTensor(),
            transforms.Normalize([0.5]*self.channels_A, [0.5]*self.channels_A)
        ]

        if post_transform_A is not None:
            self.post_transform_A += post_transform_A

        self.post_transform_A = transforms.Compose(self.post_transform_A)

        self.post_transform_B = [
            transforms.ToTensor(),
            transforms.Normalize([0.5]*self.channels_B, [0.5]*self.channels_B)
        ]     

        if post_transform_B is not None:
            self.post_transform_B += post_transform_B

        self.post_transform_B = transforms.Compose(self.post_transform_B)


    def __getitem__(self, index):
        file_name = self.image_files[index % len(self.image_files)]

        A_path = os.path.join(self.folder_A, file_name)
        B_path = os.path.join(self.folder_B, file_name)

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        if self.channels_A == 1:
            A_img = VF.rgb_to_grayscale(A_img)
        if self.channels_B == 1:
            B_img = VF.rgb_to_grayscale(B_img)

        # apply image transformation
        if self.pre_transform_A is not None:
            A_img = self.pre_transform_A(A_img)
        if self.pre_transform_B is not None:
            B_img = self.pre_transform_B(B_img)

        _, h_A, w_A = VF.get_dimensions(A_img)
        _, h_B, w_B = VF.get_dimensions(B_img)

        if h_A<self.img_size or w_A<self.img_size:
            if h_A<w_A:
                A_img = VF.resize(A_img, (int(w_A*self.img_size/h_A), self.img_size))
                B_img = VF.resize(B_img, (int(w_A*self.img_size/h_A), self.img_size))
            else:
                A_img = VF.resize(A_img, (self.img_size, int(h_A*self.img_size/w_A)))
                B_img = VF.resize(B_img, (self.img_size, int(h_A*self.img_size/w_A)))

            _, h_A, w_A = VF.get_dimensions(A_img)
            _, h_B, w_B = VF.get_dimensions(B_img)
                
        #the random transforms should be synchronized for the two sides
        if self.img_size != h_A or self.img_size != w_A: 
            available_0 = h_A - self.img_size
            available_1 = w_A - self.img_size

            start_0 = random.randint(0, available_0)
            start_1 = random.randint(0, available_1)

            A_img = VF.crop(A_img, start_0, start_1, self.img_size, self.img_size)
            B_img = VF.crop(B_img, start_0, start_1, self.img_size, self.img_size)

        #flip both with 50% probability
        if random.random() > 0.5:
            A_img = VF.hflip(A_img)
            B_img = VF.hflip(B_img)

        A_tensor = self.post_transform_A(A_img)
        B_tensor = self.post_transform_B(B_img)

        return {'A': A_tensor, 'B': B_tensor}

    def __len__(self):
        return len(self.image_files)