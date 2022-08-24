import kornia.utils.draw as draw
from kornia.geometry.conversions import denormalize_pixel_coordinates
from  kornia.geometry.transform import warp_perspective
from kornia.utils import create_meshgrid
from einops import rearrange

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms

from models import Homography, NeuralRenderer, SineLayer, Siren
import copy

import cv2 as cv

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(15, 15))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
def get_random_Warp():
    while True:
        w = torch.rand(8)*0.3

        w[0] = torch.rand(1)*2 -1
        w[1] = torch.rand(1)*2 - 1

        w[5] = torch.rand(1)*0.4+0.3
        w[4] = -0.5*w[5]

        
        H = Homography(w)

        corners = torch.tensor([[-1.0, -1.0, 1],
                               [1.0, -1.0, 1],
                               [1.0, 1.0, 1],
                               [-1.0, 1.0, 1]])
        corners_H = H(corners.T).T
        corners_H = corners_H/corners_H[:,2:] # normalize homogeneous coordinates
        if torch.all((corners_H[:,:2] < 1) & (corners_H[:,:2] > -1)):
            return H, w, corners_H

def draw_patches(img, Hs):
    with torch.no_grad():
        C,H,W = img.size()
        img_numpy = rearrange(img*255,'C H W -> H W C')
        img_numpy = img_numpy.cpu().numpy().astype(np.uint8)

        for T in Hs:
            T = copy.deepcopy(T).cpu()
            corners = torch.tensor([[-1.0, -1.0, 1],
                                   [1.0, -1.0, 1],
                                   [1.0, 1.0, 1],
                                   [-1.0, 1.0, 1]])
            corners_H = T(corners.T).T
            corners_H = corners_H/corners_H[:,2:] # normalize homogeneous coordinates
            corners_H = denormalize_pixel_coordinates(corners_H[:,:2], H, W).detach().numpy().astype(np.int32)
            
            color = tuple(np.random.randint(0,256,3))    
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
            img_numpy = cv.polylines(img_numpy, [corners_H], True, color, 3)
        
        transform = transforms.ToTensor()
        
        return transform(img_numpy)

    
def get_random_Patch(img, h=500, w=500):
    with torch.no_grad():
        img = img.cpu()
        if len(img.size())==3:
            img = img.unsqueeze(0)

        # get random homography
        T, weights, corners_H = get_random_Warp()

        return get_Patch(img, T, h, w), T
        B,C,H,W = img.size()
        x = create_meshgrid(h,w).squeeze()
        ones = torch.ones((h,w,1))
        x = torch.concat([x,ones], dim=2)
        x = rearrange(x,'H W C -> (H W) C')
        x_hom = (T(x.T)).T
        x_euc = x_hom/x_hom[:,2:]    
        x_euc = x_euc[:,:2] 
        x_euc = rearrange(x_euc, '(H W) C -> H W C', H=h,W=w).unsqueeze(0) # (1, H, W, 2)
        y = F.grid_sample(img, x_euc, align_corners=True).squeeze()
    
        return y, T
    

def get_Patch(img, T, h=500, w=500):
    with torch.no_grad():
        img = img.cpu()
        if len(img.size())==3:
            img = img.unsqueeze(0)
    
        B,C,H,W = img.size()
        x = create_meshgrid(h,w).squeeze()
        ones = torch.ones((h,w,1))
        x = torch.concat([x,ones], dim=2)
        x = rearrange(x,'H W C -> (H W) C')
        x_hom = (T(x.T)).T
        x_euc = x_hom/x_hom[:,2:]    
        x_euc = x_euc[:,:2] 
        x_euc = rearrange(x_euc, '(H W) C -> H W C', H=h,W=w).unsqueeze(0) # (1, H, W, 2)
        y = F.grid_sample(img, x_euc, align_corners=True).squeeze()
    
        return y