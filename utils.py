import numpy as np
import random
import os

from PIL import Image
import cv2

import torch
import torch.nn as nn

from skimage.measure import compare_psnr, compare_ssim

def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

"""
Ref: https://github.com/Penn000/SpA-GAN_for_cloud_removal/blob/master/utils.py
"""
def get_heatmap(img):
    if len(img.shape) == 3:
        b, h, w = img.shape
        heat = np.zeros((b,3,h,w)).astype('uint8')
        for i in range(b):
            heat[i,:,:,:] = np.transpose(cv2.applyColorMap(img[i,:,:],cv2.COLORMAP_JET),(2,0,1))
    else:
        b, c, h, w = img.shape
        heat = np.zeros((b,3,h,w)).astype('uint8')
        for i in range(b):
            heat[i,:,:,:] = np.transpose(cv2.applyColorMap(img[i,0,:,:],cv2.COLORMAP_JET),(2,0,1))
    return heat

def save_heatmap(cloud_mask, save_path, image_name):
    for idx, mask_ in enumerate(cloud_mask):
        mask = mask_.cpu().numpy() * 255
        mask = get_heatmap(mask.astype('uint8'))[0]
        mask = np.transpose(mask, (1,2,0))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(save_path, image_name + f'_maskA{idx+1}.png'), mask)

"""
Ref: https://github.com/ameraner/dsen2-cr/blob/main/Code/tools/dataIO.py
"""
def get_rgb(image):
    image = image.mul(0.5).add_(0.5)
    image = image.squeeze()
    image = image.mul(10000).add_(0.5).clamp_(0, 10000)
    image = image.permute(1, 2, 0).cpu().detach().numpy()
    image = image.astype(np.uint16)

    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]

    r = np.clip(r, 0, 2000)
    g = np.clip(g, 0, 2000)
    b = np.clip(b, 0, 2000)

    rgb = np.dstack((r, g, b))
    rgb = rgb - np.nanmin(rgb)

    # treat saturated images, scale values
    if np.nanmax(rgb) == 0:
        rgb = 255 * np.ones_like(rgb)
    else:
        rgb = 255 * (rgb / np.nanmax(rgb))

    # replace nan values before final conversion
    rgb[np.isnan(rgb)] = np.nanmean(rgb)
    rgb = rgb.astype(np.uint8)

    return rgb

def save_image(image, save_path, image_name):
    image = Image.fromarray(image)
    image.save(os.path.join(save_path, image_name))

def psnr_ssim_cal(cloudfree, predict):
    psnr = compare_psnr(cloudfree, predict)
    ssim = compare_ssim(cloudfree, predict, multichannel = True, gaussian_weights = True, use_sample_covariance = False, sigma = 1.5)
    return psnr, ssim

def PSNR_SSIM(cloudless, predict, save_path):
    predict_rgb = get_rgb(predict, save_path + 'fake_B.png')
    cloudless_rgb = get_rgb(cloudless, save_path + 'real_B.png')
    psnr, ssim = psnr_ssim_cal(cloudless_rgb, predict_rgb)

    return psnr, ssim

"""
Ref: https://github.com/ermongroup/STGAN/blob/master/models/networks.py
"""
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real, noise):

        if target_is_real:
            target_tensor = self.real_label
            target_tensor = target_tensor.expand_as(prediction).clone()
            if noise:
                real_label_noise = (torch.rand(prediction.shape[0], 1, 1, 1) - 0.5) * 6.0
                real_label_noise = real_label_noise.cuda()
                target_tensor += real_label_noise
        else:
            target_tensor = self.fake_label
            target_tensor = target_tensor.expand_as(prediction).clone()
            if noise:
                fake_label_noise = torch.rand(prediction.shape[0], 1, 1, 1) * 3.0
                fake_label_noise = fake_label_noise.cuda()
                target_tensor += fake_label_noise
        return target_tensor

    def __call__(self, prediction, target_is_real, noise):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real, noise)
            loss = self.loss(prediction, target_tensor)
        return loss
