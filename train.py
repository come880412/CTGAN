import numpy as np
from torch.serialization import save
from model.CTGAN import CTGAN_Generator, CTGAN_Discriminator
from dataset import Sen2_MTC
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import argparse
import torch
import cv2
import tqdm
from tensorboardX import SummaryWriter
from utils import *
import os
import cv2
from torch.optim.lr_scheduler import LambdaLR
import warnings
warnings.filterwarnings("ignore")

def train(opt, model_GEN, model_DIS, cloud_detection_model, optimizer_G, optimizer_D, val_loader):
    writer = SummaryWriter('runs/%s' % opt.dataset_name)

    cuda = True if torch.cuda.is_available() else False

    # Define loss functions
    criterionGAN = GANLoss(opt.gan_mode)
    criterionL1 = torch.nn.L1Loss()
    criterionMSE = nn.MSELoss()

    # Use pretrained model
    if opt.gen_path:
        print('loading pre-trained model')
        model_GEN.load_state_dict(torch.load(opt.gen_path))
        model_DIS.load_state_dict(torch.load(opt.dis_path))

    # Use GPU
    if cuda:
        criterionGAN = criterionGAN.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
        cloud_detection_model = cloud_detection_model.cuda()
        model_GEN = model_GEN.cuda()
        model_DIS = model_DIS.cuda()

    """lr_scheduler"""
    def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - opt.lr_decay) / float(opt.lr_decay + 1)
            return lr_l
    scheduler_G = LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = LambdaLR(optimizer_D, lr_lambda=lambda_rule)
    
    """training"""
    train_update = 0
    psnr_max = 0.
    ssim_max = 0.
    print('Start training!')
    lr = optimizer_G.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)
    for epoch in range(opt.initial_epoch, opt.n_epochs):
        train_data = Sen2_MTC(opt, 'train')
        train_loader = DataLoader(train_data, batch_size=opt.batch_size,shuffle=True, num_workers=opt.n_cpu)
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, opt.n_epochs), unit=" step")
        model_GEN.train()
        model_DIS.train()
        cloud_detection_model.eval()
        set_requires_grad(cloud_detection_model, False)
        L1_total = 0
        for real_A, real_B, _ in train_loader:
            real_A[0], real_A[1], real_A[2], real_B = real_A[0].cuda(), real_A[1].cuda(), real_A[2].cuda(), real_B.cuda()
            M0, _, _ = cloud_detection_model(real_A[0])
            M1, _, _ = cloud_detection_model(real_A[1])
            M2, _, _ = cloud_detection_model(real_A[2])

            real_A_combined = torch.cat((real_A[0], real_A[1], real_A[2]), 1).cuda()
            """forward generator"""
            fake_B, cloud_mask, aux_pred = model_GEN(real_A)

            """update Discriminator"""
            set_requires_grad(model_DIS, True)
            optimizer_D.zero_grad()

            # Fake 
            fake_AB = torch.cat((real_A_combined, fake_B), 1)
            pred_fake = model_DIS(fake_AB.detach())
            loss_D_fake = criterionGAN(pred_fake, False)

            # Real
            real_AB = torch.cat((real_A_combined, real_B), 1)
            pred_real = model_DIS(real_AB)
            loss_D_real = criterionGAN(pred_real, True)

            # Combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optimizer_D.step()

            """update generator"""
            optimizer_G.zero_grad()
            set_requires_grad(model_DIS, False)

            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((real_A_combined, fake_B), 1)
            pred_fake = model_DIS(fake_AB)
            loss_G_GAN = criterionGAN(pred_fake, True)

            # Second, G(A) = B
            loss_G_L1 = criterionL1(fake_B, real_B) * opt.lambda_L1
            L1_total += loss_G_L1.item()

            # combine loss and calculate gradients
            loss_g_att1 = criterionMSE(cloud_mask[0][:,0,:,:], M0[:,0,:,:])
            loss_g_att2 = criterionMSE(cloud_mask[1][:,0,:,:], M1[:,0,:,:])
            loss_g_att3 = criterionMSE(cloud_mask[2][:,0,:,:], M2[:,0,:,:])
            loss_g_att = loss_g_att1 + loss_g_att2 + loss_g_att3
            if opt.aux_loss:
                loss_G_aux = (criterionL1(aux_pred[0], real_B) + criterionL1(aux_pred[1], real_B) + criterionL1(aux_pred[2], real_B)) * opt.lambda_aux
                loss_G = loss_G_GAN + loss_G_L1 + loss_g_att + loss_G_aux
            else:
                loss_G = loss_G_GAN + loss_G_L1 + loss_g_att
            loss_G.backward()
            optimizer_G.step()

            writer.add_scalar('training_G_GAN', loss_G_GAN, train_update)
            writer.add_scalar('training_G_L1', loss_G_L1, train_update)
            writer.add_scalar('training_D_real', loss_D_real, train_update)
            writer.add_scalar('training_D_fake', loss_D_fake, train_update)
            writer.add_scalar('training_D_fake', loss_g_att, train_update)

            pbar.update()
            pbar.set_postfix(
            G_GAN=f"{loss_G_GAN:.4f}",
            G_L1 = f"{loss_G_L1:.4f}",
            G_L1_total=f"{L1_total:.4f}",
            D_real=f"{loss_D_real:.4f}",
            D_fake=f"{loss_D_fake:.4f}"
            )
            train_update += 1
        pbar.close()
        """validation"""
        psnr, ssim = valid(opt, model_GEN, val_loader, writer, epoch)

        if psnr_max < psnr:
          psnr_max = psnr
          torch.save(model_GEN.state_dict(), '%s/%s/%s/G_epoch%d_PSNR%.3f.pth' % (opt.save_model_path, opt.dataset_name, opt.data_mode, epoch, psnr_max))
          torch.save(model_DIS.state_dict(), '%s/%s/%s/D_epoch%d_PSNR%.3f.pth' % (opt.save_model_path, opt.dataset_name, opt.data_mode, epoch, psnr_max))
          print('save model!')
        if ssim_max < ssim:
          ssim_max = ssim
          torch.save(model_GEN.state_dict(), '%s/%s/%s/G_epoch%d_SSIM%.3f.pth' % (opt.save_model_path, opt.dataset_name, opt.data_mode, epoch, ssim_max))
          torch.save(model_DIS.state_dict(), '%s/%s/%s/D_epoch%d_SSIM%.3f.pth' % (opt.save_model_path, opt.dataset_name, opt.data_mode, epoch, ssim_max))
          print('save model!')
      
        scheduler_D.step()
        scheduler_G.step()
        lr = optimizer_G.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
    print('best PSNR:', psnr_max)
    print('best SSIM:', ssim_max)

def valid(opt, model_GEN, val_loader, writer, epoch):
    model_GEN.eval()
    criterionL1 = torch.nn.L1Loss().cuda()
    psnr_list = []
    ssim_list = []
    total_loss = 0
    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="valid", unit=" step")
    for real_A, real_B, image_path in val_loader:
        with torch.no_grad():
            image_name = image_path[0].split('/')[-1].split('.')[0]
            real_A[0], real_A[1], real_A[2], real_B = real_A[0].cuda(), real_A[1].cuda(), real_A[2].cuda(), real_B.cuda()
            fake_B, cloud_mask, _ = model_GEN(real_A)
            loss = criterionL1(fake_B, real_B)
            
            save_path = '%s/%s/%s' % (opt.predict_image_path, opt.data_mode, image_name)
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
    writer.add_scalar('validation_PSNR', psnr, epoch)
    writer.add_scalar('validation_SSIM', ssim, epoch)
    pbar.set_postfix(loss_val=f"{total_loss:.4f}", psnr=f"{psnr:.3f}", ssim=f"{ssim:.3f}")
    pbar.close()
    return psnr, ssim
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """train_model"""
    parser.add_argument("--n_epochs", type=int, default=120, help="number of epochs you want to train")
    parser.add_argument("--initial_epoch", type=int, default=0, help="Start epoch")
    parser.add_argument("--lr_decay", type=int, default=60, help="Start to lr_decay")
    parser.add_argument("--gan_mode", type=str, default='lsgan', help="gan mode you want to use(lsgan/vanilla)")
    parser.add_argument("--optimizer", type=str, default='Adam', help="optimizer you want to use(Adam/SGD)")
    parser.add_argument("--gen_path", type=str, default='./checkpoints/CTGAN-Sen2_MTC/G_epoch97_PSNR21.259.pth', help="path to the model of generator")
    parser.add_argument("--dis_path", type=str, default='./checkpoints/CTGAN-Sen2_MTC/D_epoch97_PSNR21.259.pth', help="path to the model of discriminator")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")   
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    """base_options"""
    parser.add_argument("--cloud_model_path", type=str, default='./checkpoints/Feature_Extrator_FS2.pth', help="path to cloud_detection_model")
    parser.add_argument("--train_path", type=str, default='../../../dataset/Sen2_MTC/train.txt', help="path to train.txt")
    parser.add_argument("--val_path", type=str, default='../../../dataset/Sen2_MTC/val.txt', help="path to val.txt")
    parser.add_argument("--dataset_path", type=str, default='../../../dataset/Sen2_MTC/data_crop', help="path to image folder")
    parser.add_argument("--data_mode", type=str, default='val', help="use (val/test) dataset")
    parser.add_argument("--data_augmentation", type=int, default=1, help="whether to do data augmentation(1/0)")
    parser.add_argument("--in_channel", type=int, default=4, help="the number of input channels")
    parser.add_argument("--out_channel", type=int, default=4, help="the number of output channels")
    parser.add_argument("--image_size", type=int, default=256, help="crop size")
    parser.add_argument("--aux_loss", type=int, default=1, help="whether use auxiliary loss(1/0)")
    parser.add_argument("--predict_image_path", type=str, default='./image_out', help="name of the saved data path")
    parser.add_argument("--save_model_path", type=str, default='./checkpoints', help="name of the saved model path")
    parser.add_argument("--dataset_name", type=str, default='CTGAN_Sen2_MTC', help="name of the dataset")
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--lambda_aux', type=float, default=50.0, help='weight for aux loss')
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id")
    opt = parser.parse_args()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    print(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    os.makedirs("%s/%s/%s" % (opt.save_model_path, opt.dataset_name, opt.data_mode), exist_ok=True)
    os.makedirs("%s/%s" % (opt.predict_image_path, opt.data_mode) , exist_ok=True)

    random_seed_general = 412
    random.seed(random_seed_general)  # random package

    val_data = Sen2_MTC(opt, opt.data_mode)
    val_loader = DataLoader(val_data, batch_size=1,shuffle=False, num_workers=opt.n_cpu)

    print('Load cloud_detection_model')
    cloud_detection_model = CTGAN_Generator(image_size=224)
    cloud_detection_model = cloud_detection_model.feature_extractor
    cloud_detection_model.load_state_dict(torch.load(opt.cloud_model_path))
    
    print('Load CTGAN model')
    GEN = CTGAN_Generator(image_size=opt.image_size)
    DIS = CTGAN_Discriminator()
    # print(GEN, DIS)
    
    if opt.optimizer == 'Adam':
        optimizer_G = torch.optim.Adam(GEN.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(DIS.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    if opt.optimizer == 'SGD':
        optimizer_G = torch.optim.SGD(GEN.parameters(), lr=opt.lr, momentum=0.9)
        optimizer_D = torch.optim.SGD(DIS.parameters(), lr=opt.lr, momentum=0.9)
    
    train(opt, GEN, DIS, cloud_detection_model, optimizer_G, optimizer_D, val_loader)