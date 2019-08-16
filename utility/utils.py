"""
Script containing the utilities needed for the
program. 
"""
import os
import copy

import torch
import torch.hub
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms
import cv2
import matplotlib.cm as cm
import numpy as np 


def device_signature(cuda_flag):
    cuda = cuda_flag
    if cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        current_device = torch.cuda.current_device()
        print("Computing Device:  {:}".format(torch.cuda.get_device_name(current_device)))
    else: 
        device = torch.device("cpu")
        current_device = device
        print("Computing Device:  {:}".format(current_device))
    
    return device

def access_classlabels():
    classlabels = []
    with open('./accessories/synset_words.txt', 'r') as infile:
        data = infile.readlines()
        for line in data:
            line = line.strip().split(" ", 1)[1]
            category = line.split(", ", 1)[0].replace(" ", "_")
            classlabels.append(category)
    return classlabels

def write_gradient(filename, gradient):
    grad_cpu = gradient.cpu()
    grad_np = grad_cpu.numpy().transpose(1, 2, 0)
    grad_minval = grad_np.min()
    grad_maxval = grad_np.max()
    grad_np -= grad_minval
    grad_np /= grad_maxval
    grad_np *= 255.0
    cv2.imwrite(filename, np.uint8(grad_np)) 
    
def write_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))
    
def write_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)
    
def access_models():
    model_names = [name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name])]
    model_names = sorted(model_names)
    return model_names
  
def image_preprocessing(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image
