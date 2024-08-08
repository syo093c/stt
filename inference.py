from PIL import Image
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichModelSummary
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
import lightning as L
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from torch.nn import Sigmoid
#==========
import sys
#sys.path.append('.')
from depth_anything_v2.dpt import DepthAnythingV2
from dataset import STDataset
import ipdb
from torch.optim import AdamW
import matplotlib.pyplot as plt

from torchvision.transforms import Compose
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2
from torch.nn import Sigmoid

def main():
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs['vits']})
    model.load_state_dict(torch.load('model_140000.0.pth'))
    model=model.cuda()
    model.eval()

    data=cv2.imread('a.jpeg')

    input_size=518
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    #shapes= data.shape[:2]#h,w
    image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({'image': image})['image']
    image = torch.from_numpy(image)
    image = image.unsqueeze(0).cuda()

    out=model(image)
    out=Sigmoid()(out)

    out=out.squeeze(0).detach().cpu().numpy()*255.
    out = np.transpose(out, (1, 2, 0))

    image=Image.fromarray(out.astype(np.uint8))
    image.save('out.png')



if __name__ == "__main__":
    main()
