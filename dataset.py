import torch
from torch.utils.data import Dataset,DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
import ipdb
import rasterio
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import albumentations as A
import cv2
from torchvision.transforms import Compose
from depth_anything_v2.dinov2 import DINOv2
from depth_anything_v2.util.blocks import FeatureFusionBlock, _make_scratch
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
import ipdb



class STDataset(Dataset):
    def __init__(self,data_path,label_path,data_transforms=None):
        self.data=sorted(glob(data_path+'**/*'))
        self.label=sorted(glob(label_path+'**/*'))
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index,input_size=518):
        data=cv2.imread(self.data[index])
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

        #------------------------
        label_transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            #NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        label=cv2.imread(self.label[index])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB) / 255.0
        label = label_transform({'image': label})['image']
        label = torch.from_numpy(label)

        train_batch={'data':image,"label":label}
        return train_batch

#class DuckTrainDataset(Dataset):
#
#class DuckValDataset(Dataset):
#
#class DuckTestDataset(Dataset):
#