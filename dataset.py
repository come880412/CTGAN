from torch.utils.data import Dataset
# import rasterio
import os
import tifffile as tiff
from PIL import Image
import numpy as np
import torch
from utils import heatmap
import cv2

class Sen2_MTC(Dataset):
    def __init__(self, opt, mode):
        self.filepair = []
        if mode == 'train':
            self.data_augmentation = opt.data_augmentation
            self.tile_list = np.loadtxt(opt.train_path, dtype=str)
        elif mode == 'val':
            self.data_augmentation = 0
            self.tile_list = np.loadtxt(opt.val_path, dtype=str)
        elif mode == 'test':
            self.data_augmentation = 0
            self.tile_list = np.loadtxt(opt.test_path, dtype=str)
        for tile_path in self.tile_list:
            for image_file in os.listdir(os.path.join(tile_path, 'cloudless')):
                img_name = image_file.split('.')[0]
                image_cloud_path0 = tile_path + '/cloud/' + img_name  + '_0.tif'
                image_cloud_path1 = tile_path + '/cloud/' + img_name  + '_1.tif'
                image_cloud_path2 = tile_path + '/cloud/' + img_name  + '_2.tif'
                image_cloudless_path = tile_path + '/cloudless/' + image_file
                self.filepair.append([image_cloud_path0, image_cloud_path1, image_cloud_path2, image_cloudless_path])
        if self.data_augmentation:
            self.augment_rotation_param = np.random.randint(0, 4, len(self.filepair))
            self.augment_flip_param = np.random.randint(0, 3, len(self.filepair))
        self.index = 0

    def __getitem__(self, index):
        cloud_image_path0, cloud_image_path1, cloud_image_path2 = self.filepair[index][0], self.filepair[index][1], self.filepair[index][2]
        cloudless_image_path = self.filepair[index][3]
        image_cloud0 = self.image_read(cloud_image_path0, 0)
        image_cloud1 = self.image_read(cloud_image_path1, 1)
        image_cloud2 = self.image_read(cloud_image_path2, 2)
        image_cloudless = self.image_read(cloudless_image_path, 3)

        return [image_cloud0, image_cloud1, image_cloud2], image_cloudless, cloudless_image_path
        
    def __len__(self):
        return len(self.filepair)

    def image_read(self, image_path, index):
        img = tiff.imread(image_path)
        img = img/1.0
        img = img.transpose((2, 0, 1))
        if self.data_augmentation:
            if not self.augment_flip_param[self.index//4] == 0:
                img = np.flip(img, self.augment_flip_param[self.index//4])
            if not self.augment_rotation_param[self.index//4] == 0:
                img = np.rot90(img, self.augment_rotation_param[self.index//4], (1, 2))
            self.index += 1

        image = torch.from_numpy((img.copy()))
        image = image.float()/10000.0
        mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)
        
        return image