import numpy as np
# import rasterio
import os
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr, compare_ssim
from PIL import Image
import torch
import torch.nn as nn
import cv2
from torch.autograd import Variable

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def heatmap(img):
    if len(img.shape) == 3:
        b,h,w = img.shape
        heat = np.zeros((b,3,h,w)).astype('uint8')
        for i in range(b):
            heat[i,:,:,:] = np.transpose(cv2.applyColorMap(img[i,:,:],cv2.COLORMAP_JET),(2,0,1))
    else:
        b,c,h,w = img.shape
        heat = np.zeros((b,3,h,w)).astype('uint8')
        for i in range(b):
            heat[i,:,:,:] = np.transpose(cv2.applyColorMap(img[i,0,:,:],cv2.COLORMAP_JET),(2,0,1))
    return heat

def save_heatmap(cloud_mask, save_path):
    for idx, mask_ in enumerate(cloud_mask):
        mask = mask_.cpu().numpy()[0] * 255
        mask = heatmap(mask.astype('uint8'))[0]
        mask = np.transpose(mask, (1,2,0))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path + '_maskA%d.png' % (idx+1), mask)

def TIF_to_RGB(img, save_path):
    image = img.mul(0.5).add_(0.5)
    image = image.squeeze()
    image = image.mul(10000).add_(0.5).clamp_(0, 10000)
    image = image.permute(1, 2, 0).cpu().detach().numpy()
    rgb_image = get_rgb(image.astype(np.uint16), save_path)
    return rgb_image

def save_image(image, save_path):
    for idx, real in enumerate(image):
        real_A = real[0]
        _ = TIF_to_RGB(real_A, save_path + '_real_A%d.png' % (idx + 1))

def val_pnsr_and_ssim(cloudfree, predict):
    psnr = compare_psnr(cloudfree, predict)
    ssim = compare_ssim(cloudfree, predict, multichannel = True, gaussian_weights = True, use_sample_covariance = False, sigma = 1.5)
    return psnr, ssim

def PSNR_SSIM(cloudless, predict, save_path):
    # predict = np.array(tiff.imread('./image_out/valid_combined/%s/fake_B.tif' % (site_name)), np.float32)
    # print(save_path + 'fake_B.png')
    predict_rgb = get_rgb(predict, save_path + 'fake_B.png')
    cloudless_rgb = get_rgb(cloudless, save_path + 'real_B.png')
    psnr, ssim = val_pnsr_and_ssim(cloudless_rgb, predict_rgb)

    return psnr, ssim

def get_rgb(image, save_path):
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

    save_image = Image.fromarray(rgb)
    save_image.save(save_path)
    return np.array(save_image)

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
