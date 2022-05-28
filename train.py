import numpy as np
from torch.serialization import save
from torch.optim.lr_scheduler import CosineAnnealingLR
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
import warnings
warnings.filterwarnings("ignore")

cuda = True if torch.cuda.is_available() else False
def train(opt, model_GEN, model_DIS, cloud_detection_model, optimizer_G, optimizer_D, train_loader, val_loader):
    writer = SummaryWriter('runs/%s' % opt.dataset_name)
    
    # Define loss functions
    noise = opt.label_noise
    criterionGAN = GANLoss(opt.gan_mode)
    criterionL1 = torch.nn.L1Loss()
    criterionMSE = nn.MSELoss()

    # Use GPU
    if cuda:
        criterionGAN = criterionGAN.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
        cloud_detection_model = cloud_detection_model.cuda()
        model_GEN = model_GEN.cuda()
        model_DIS = model_DIS.cuda()

    """lr_scheduler"""
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=opt.n_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=opt.n_epochs, eta_min=1e-6)
    
    """training"""
    train_update = 0
    psnr_max = 0.
    ssim_max = 0.

    print('Start training!')
    for epoch in range(opt.n_epochs):
        model_GEN.train()
        model_DIS.train()

        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, opt.n_epochs), unit=" step")
        
        lr = optimizer_G.param_groups[0]['lr']
        print('\nlearning rate = %.7f' % lr)

        L1_total = 0
        for real_A, real_B, _ in train_loader:
            real_A[0], real_A[1], real_A[2], real_B = real_A[0].cuda(), real_A[1].cuda(), real_A[2].cuda(), real_B.cuda()

            with torch.no_grad():
                M0, _, _ = cloud_detection_model(real_A[0])
                M1, _, _ = cloud_detection_model(real_A[1])
                M2, _, _ = cloud_detection_model(real_A[2])
            M = [M0, M1, M2]

            real_A_combined = torch.cat((real_A[0], real_A[1], real_A[2]), 1).cuda()

            """forward generator"""
            fake_B, cloud_mask, aux_pred = model_GEN(real_A)

            """update Discriminator"""
            set_requires_grad(model_DIS, True)
            optimizer_D.zero_grad()

            # Fake 
            fake_AB = torch.cat((real_A_combined, fake_B), 1)
            pred_fake = model_DIS(fake_AB.detach())
            loss_D_fake = criterionGAN(pred_fake, False, noise)

            # Real
            real_AB = torch.cat((real_A_combined, real_B), 1)
            pred_real = model_DIS(real_AB)
            loss_D_real = criterionGAN(pred_real, True, noise)

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
            loss_G_GAN = criterionGAN(pred_fake, True, noise)

            # Second, G(A) = B
            loss_G_L1 = criterionL1(fake_B, real_B) * opt.lambda_L1
            L1_total += loss_G_L1.item()

            # combine loss and calculate gradients
            loss_g_att = 0
            for i in range(len(cloud_mask)):
                loss_g_att += criterionMSE(cloud_mask[i][:, 0, :, :], M[i][:, 0, :, :])

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
        psnr, ssim = valid(opt, model_GEN, val_loader, criterionL1, writer, epoch)

        if psnr_max < psnr:
          psnr_max = psnr
          torch.save(model_GEN.state_dict(), os.path.join(opt.save_model_path, opt.opt.dataset_name, 'G_best_PSNR.pth'))
          torch.save(model_DIS.state_dict(), os.path.join(opt.save_model_path, opt.opt.dataset_name, 'D_best_PSNR.pth'))
          print('Save model!')
        if ssim_max < ssim:
          ssim_max = ssim
          torch.save(model_GEN.state_dict(), os.path.join(opt.save_model_path, opt.opt.dataset_name, 'G_best_SSIM.pth'))
          torch.save(model_DIS.state_dict(), os.path.join(opt.save_model_path, opt.opt.dataset_name, 'D_best_SSIM.pth'))
          print('Save model!')
      
        scheduler_D.step()
        scheduler_G.step()

    print('Best PSNR: %.3f | Best SSIM: %.3f' % (psnr_max, ssim_max))

def valid(opt, model_GEN, val_loader, criterionL1, writer, epoch):
    model_GEN.eval()

    psnr_list = []
    ssim_list = []
    total_loss = 0

    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="Valid[%d/%d]" % (epoch, opt.n_epochs), unit=" step")
    save_path = os.path.join(opt.predict_image_path, 'val')
    with torch.no_grad():
        for (real_A, real_B, image_names) in val_loader:
            real_A[0], real_A[1], real_A[2], real_B = real_A[0].cuda(), real_A[1].cuda(), real_A[2].cuda(), real_B.cuda()
            fake_B, cloud_mask, _ = model_GEN(real_A)

            loss = criterionL1(fake_B, real_B)
            
            for batch in range(opt.batch_size):
                image_name = image_names[batch]
                output, label = fake_B[batch], real_B[batch]
                input_1, input_2, input_3 = real_A[0][batch], real_A[1][batch], real_A[2][batch]
                input_1, input_2, input_3 = get_rgb(input_1), get_rgb(input_2), get_rgb(input_3)

                for idx, real_img in enumerate([input_1, input_2, input_3]):
                    save_image(real_img, save_path, image_name + f'_real_A{idx + 1}.png')

                save_heatmap([cloud_mask[0][batch], cloud_mask[1][batch], cloud_mask[2][batch]], save_path, image_name)

                output_rgb, label_rgb = get_rgb(output), get_rgb(label)
                save_image(output_rgb, save_path, image_name + '_fake_B.png')
                save_image(label_rgb, save_path, image_name + '_real_B.png')

                psnr, ssim = psnr_ssim_cal(label_rgb, output_rgb)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

            total_loss += loss.item()
            pbar.update()
            pbar.set_postfix(
                loss_val=f"{total_loss:.4f}"
            )
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
    """Path"""
    parser.add_argument("--root", type=str, default='../dataset/Sen2_MTC', help="Path to dataset")
    parser.add_argument("--cloud_model_path", type=str, default='./Pretrain/Feature_Extrator_FS2.pth', help="path to feature extractor model")
    parser.add_argument("--save_model_path", type=str, default='./checkpoints', help="Path to save model")
    parser.add_argument("--dataset_name", type=str, default='CTGAN_Sen2_MTC', help="name of the dataset")
    parser.add_argument("--predict_image_path", type=str, default='./image_out', help="Path to save predicted images")
    parser.add_argument("--load_gen", type=str, default='./Pretrain/CTGAN-Sen2_MTC/G_epoch97_PSNR21.259.pth', help="path to the model of generator")
    parser.add_argument("--load_dis", type=str, default='./Pretrain/CTGAN-Sen2_MTC/D_epoch97_PSNR21.259.pth', help="path to the model of discriminator")

    """Parameters"""
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--gan_mode", type=str, default='lsgan', help="Which gan mode(lsgan/vanilla)")
    parser.add_argument("--optimizer", type=str, default='Adam', help="optimizer you want to use(Adam/SGD)")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")   
    parser.add_argument("--workers", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--lambda_aux', type=float, default=50.0, help='weight for aux loss')
    parser.add_argument("--in_channel", type=int, default=4, help="the number of input channels")
    parser.add_argument("--out_channel", type=int, default=4, help="the number of output channels")
    parser.add_argument("--image_size", type=int, default=256, help="crop size")
    parser.add_argument("--aux_loss", action='store_true', help="whether use auxiliary loss(1/0)")
    parser.add_argument("--label_noise", action='store_true', help="whether to add noise on the label of gan training")

    """base_options"""
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id")
    parser.add_argument("--manual_seed", type=int, default=2022, help="random_seed you want")

    opt = parser.parse_args()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    print(opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    
    os.makedirs(os.path.join(opt.save_model_path, opt.dataset_name), exist_ok=True)
    os.makedirs(os.path.join(opt.predict_image_path, 'val') , exist_ok=True)
    set_seed(opt.manual_seed)

    train_data = Sen2_MTC(opt, 'train')
    val_data = Sen2_MTC(opt, mode = 'val')

    train_loader = DataLoader(train_data, batch_size=opt.batch_size,shuffle=True, num_workers=opt.workers)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, drop_last=False)

    # Pretrained model on FS2 dataset
    print('Load cloud_detection_model')
    cloud_detection_model = CTGAN_Generator(image_size=224)
    cloud_detection_model = cloud_detection_model.feature_extractor
    cloud_detection_model.load_state_dict(torch.load(opt.cloud_model_path))
    cloud_detection_model.eval()
    set_requires_grad(cloud_detection_model, False)
    
    print('Load CTGAN model')
    GEN = CTGAN_Generator(image_size=opt.image_size)
    DIS = CTGAN_Discriminator()
    # print(GEN, DIS)

    # Use pretrained model
    if opt.load_gen and opt.load_dis:
        print('loading pre-trained model')
        GEN.load_state_dict(torch.load(opt.load_gen))
        DIS.load_state_dict(torch.load(opt.load_dis))
    
    if opt.optimizer == 'Adam':
        optimizer_G = torch.optim.Adam(GEN.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(DIS.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    if opt.optimizer == 'SGD':
        optimizer_G = torch.optim.SGD(GEN.parameters(), lr=opt.lr, momentum=0.9, nesterov=True)
        optimizer_D = torch.optim.SGD(DIS.parameters(), lr=opt.lr, momentum=0.9, nesterov=True)
    
    train(opt, GEN, DIS, cloud_detection_model, optimizer_G, optimizer_D, train_loader, val_loader)