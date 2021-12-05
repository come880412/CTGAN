import numpy as np
from skimage.util.arraycrop import crop
from model.CTGAN import CTGAN_Generator
from dataset import Sen2_MTC
from torch.utils.data import DataLoader
import random
import argparse
import torch
import tqdm
import cv2
from utils import *
import os
import warnings
warnings.filterwarnings("ignore")

def test_and_visualization(opt, model_GEN, test_loader):
    model_GEN.eval()
    criterionL1 = torch.nn.L1Loss().cuda()
    psnr_list = []
    ssim_list = []
    total_loss = 0
    pbar = tqdm.tqdm(total=len(test_loader), ncols=0, desc="%s_separate"%(opt.test_mode), unit=" step")
    for (real_A, real_B, image_path) in test_loader:
        with torch.no_grad():
            image_name = image_path[0].split('/')[-1].split('.')[0]
            real_A[0], real_A[1], real_A[2], real_B = real_A[0].cuda(), real_A[1].cuda(), real_A[2].cuda(), real_B.cuda()
            fake_B, cloud_mask, _ = model_GEN(real_A)
            loss = criterionL1(fake_B, real_B)
            
            save_path = '%s/%s/%s' % (opt.predict_image_path, opt.test_mode, image_name)
            save_heatmap(cloud_mask, save_path)
            save_image(real_A, save_path)

            pred_img_rgb = TIF_to_RGB(fake_B[0], save_path + '_fake_B.png')
            target_img_rgb = TIF_to_RGB(real_B[0], save_path + '_real_B.png')

            psnr, ssim = val_pnsr_and_ssim(target_img_rgb, pred_img_rgb)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            total_loss += loss.item()
            pbar.update()
            pbar.set_postfix(
            loss_val=f"{total_loss:.4f}")
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    psnr = np.mean(psnr_list)
    ssim = np.mean(ssim_list)
    pbar.set_postfix(loss_val=f"{total_loss:.4f}", psnr=f"{psnr:.3f}", ssim=f"{ssim:.3f}")
    pbar.close()
    return psnr, ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """train_model"""
    parser.add_argument("--gen_checkpoint_path", type=str, default='./checkpoints/CTGAN-Sen2_MTC/G_epoch7_PSNR21.259.pth', help="which checkpoint you want to use for generator")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    """base_options"""
    parser.add_argument("--test_mode", type=str, default='test', help="which data_mode you want to use?(val/test)")
    parser.add_argument("--val_path", type=str, default='../../../dataset/multi_temporal_Sentinel2/val.txt', help="path to txt(val.txt)")
    parser.add_argument("--test_path", type=str, default='../../../dataset/multi_temporal_Sentinel2/test.txt', help="path to txt(test.txt)")
    parser.add_argument("--in_channel", type=int, default=4, help="the number of input channels")
    parser.add_argument("--image_size", type=int, default=256, help="image size")
    parser.add_argument("--crop_size", type=int, default=256, help="crop size")
    parser.add_argument("--out_channel", type=int, default=4, help="the number of output channels")
    parser.add_argument("--predict_image_path", type=str, default='./image_out', help="name of the dataset_list")
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id")
    opt = parser.parse_args()
    random_seed_general = 412
    random.seed(random_seed_general)  # random package
    os.makedirs("%s/%s" %(opt.predict_image_path, opt.test_mode) , exist_ok=True)
    test_data = Sen2_MTC(opt, opt.test_mode)
    test_loader = DataLoader(test_data, batch_size=1,shuffle=False, num_workers=opt.n_cpu)

    """define model & optimizer"""
    GEN = CTGAN_Generator(opt.image_size)
    GEN.load_state_dict(torch.load(opt.gen_checkpoint_path))
    print('load transformer model successfully!')

    GEN = GEN.cuda()
    test_and_visualization(opt = opt, model_GEN = GEN, test_loader = test_loader)