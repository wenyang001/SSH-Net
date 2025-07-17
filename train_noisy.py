#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : train_tri.py
# @Software: PyCharm
import os
import argparse
import string
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import DPAUNet, HN
from model_ours import SSHNet
from dataset import prepare_data, Dataset 
from utils import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="SWCNN")
config = get_config('configs/config_DRDNetwork.yaml')
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers(DnCNN)")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
# parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--alpha", type=float, default=0.3, help="The opacity of the watermark")
parser.add_argument("--outf", type=str, default='./out', help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--mode_wm", type=str, default="S", help='with known alpha level (S) or blind training (B)')
parser.add_argument("--net", type=str, default="PSLNet", help='Network used in training')
parser.add_argument("--loss", type=str, default="L1", help='The loss function used for training')
parser.add_argument("--self_supervised", type=str, default="True", help='T stands for TRUE and F stands for FALSE')
parser.add_argument("--PN", type=bool, default="True", help='Whether to use perception network')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--GPU_id", type=str, default="0, 1", help='GPU_id')
opt = parser.parse_args()


# config = get_config(opt.config)
# opt.outf = config.train_model_out_path_DPAUNet
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_id

opt.outf = config['train_model_out_path_DPAUNet']

if opt.self_supervised == "True":
    model_name_3 = "n2n"
else:
    model_name_3 = "n2c"

# denoising
if opt.mode == "S":
    model_name_4 = "S" + str(int(opt.noiseL)) # known noise level
else:
    model_name_4 = "B"  # blind noise level

if opt.loss == "L2":
    model_name_5 = "L2" # known noise level
else:
    model_name_5 = "L1"

# watermarking
if opt.mode_wm == "S":
    model_name_6 = "aS"  # known alpha level
else:
    model_name_6 = "aB"  # blind alpha level


tensorboard_name = opt.net + model_name_3 + model_name_4 + model_name_5 + model_name_6 + "alpha" + str(opt.alpha)
model_name = tensorboard_name + ".pth"


def custom_collate_fn(batch, opt):
    img_train = torch.stack(batch)
    noiseL_B = [0, 55]  #known noise level
    random_img = random.randint(1, 12)
    if opt.mode_wm == "S": # 随机加入水印，random_img表示水印的编号
        imgn_train = add_watermark_noise(img_train, 40, True, random_img, alpha=opt.alpha) # 固定透明度
    else:
        imgn_train = add_watermark_noise_B(img_train, 40, True, random_img, alpha=opt.alpha) # 

    # self-supervised learning then add one more watermark --> imgn_train_2
    if opt.self_supervised == "True": # decide whether to use self-supervised learning to have imgn_train_2
        if opt.mode_wm == "S":
            imgn_train_2 = add_watermark_noise(img_train, 40, True, random_img, alpha=opt.alpha)
        else:
            imgn_train_2 = add_watermark_noise_B(img_train, 40, True, random_img, alpha=opt.alpha)
    else: # do not use self-supervised learning, then --> imgn_train_2 = img_train  (without watermark)
        imgn_train_2 = img_train

    imgn_train_mid = torch.Tensor(imgn_train) # imgn_train_mid is the same as imgn_train, with wm but without noise

    # initialize the noise as noise Gaussian
    if opt.mode == 'S': #
        if opt.noiseL != 0: # known noise level
            noise_gauss = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL / 255.)
    else:
        noise_gauss = torch.zeros(img_train.size()) # blind noise level
        stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise_gauss.size()[0])  # 0~55 noise level random level
        for n in range(noise_gauss.size()[0]):
            sizeN = noise_gauss[0, :, :, :].size()
            noise_gauss[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)

    # add noise to the imgn_train by adding noise_gauss
    if opt.noiseL == 0:
        imgn_train = torch.Tensor(imgn_train)
    else:
        imgn_train = torch.Tensor(imgn_train) + noise_gauss  # input image with noise with watermark

    imgn_train_2 = torch.Tensor(imgn_train_2)  
    return [imgn_train, imgn_train_2, imgn_train_mid, img_train]


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True, mode='color', data_path=config['train_data_path'])
    dataset_val = Dataset(train=False, mode='color', data_path=config['train_data_path'])
    loader_train = DataLoader(dataset=dataset_train, num_workers=8, batch_size=opt.batchSize, shuffle=True, pin_memory=True, collate_fn=lambda x: custom_collate_fn(x, opt))  # 4
    print("# of training samples: %d\n" % int(len(dataset_train)))

    if opt.net == "HN":
        net = HN()
    elif opt.net == "PSLNet":
        net = DPAUNet()
    elif opt.net == "SSHNet":
        net = SSHNet()  
    else:
        assert False
    
    torch.cuda.set_device(2)
    device_ids = [2,3]

    # torch.cuda.set_device(0)
    # device_ids = [0,1]

    writer = SummaryWriter("runs3/New6/" + tensorboard_name)
    model_vgg = load_froze_vgg16(device_ids)
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    # load loss function
    if opt.loss == "L2":
        criterion = nn.MSELoss(size_average=False)
    else:
        criterion = nn.L1Loss(size_average=False)

    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4, betas=(0.9, 0.99))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.5)
    noiseL_B = [0, 55]  #known noise level

    # 如果文件夹不存在，则创建它
    save_dir = os.path.join(opt.outf, model_name[:-4]) 
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    step = 0
    for epoch in range(opt.epochs):
        scheduler.step()
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        with tqdm(loader_train, desc=f'Epoch {epoch+1}/{opt.epochs}', unit='batch') as pbar:
            for i, data in enumerate(pbar):
                # training step
                model.train()
                model.zero_grad()
                optimizer.zero_grad()

                imgn_train, imgn_train_2, imgn_train_mid, img_train = [x.cuda() for x in data]
                if opt.net == "FFDNet":
                    if opt.mode == "S":
                        noise_sigma = opt.noiseL / 255.
                        noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(img_train.shape[0])]))
                    else:
                        stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise_gauss.size()[0]) 
                        noise_sigma = stdN / 255.
                        noise_sigma = torch.FloatTensor(noise_sigma)

                    noise_sigma = Variable(noise_sigma)
                    noise_sigma = noise_sigma.cuda()
                    out_train = model(imgn_train, noise_sigma)

                # output of the network
                elif opt.net == "SSHNet":
                    out_train, out_dn, out_wm = model(imgn_train)
                    feature_out_wm = model_vgg(out_wm)
                elif opt.net == "DPUNet" or opt.net == "PSLNet": # DPUNet is the same as PSLNet without Interactions
                    out_train, out_dn, out_wm = model(imgn_train)
                    feature_out_wm = model_vgg(out_wm)
                else:
                    out_train = model(imgn_train)
                
                feature_out = model_vgg(out_train)
                feature_img = model_vgg(imgn_train_2)

                if opt.net == "HN":
                    loss = (1.0 * criterion(out_train, imgn_train_2) / imgn_train.size()[
                        0] * 2) + (0.024 * criterion(feature_out, feature_img) / (feature_img.size()[0] / 2))
                elif opt.net == "SSHNet":
                    loss = (1.0 * criterion(out_train, imgn_train_2) / imgn_train.size()[
                        0] * 2) + (0.024 * criterion(feature_out, feature_img) / (feature_img.size()[0] / 2)) + (
                                1.0 * criterion(out_dn, imgn_train_mid) / imgn_train.size()[
                            0] * 2) + (
                                1.0 * criterion(out_wm, imgn_train_2) / imgn_train.size()[
                            0] * 2) + (0.024 * criterion(feature_out_wm, feature_img) / (feature_img.size()[0] / 2))
                elif opt.net == "PSLNet": # loss includes out_train, out_dn, out_wm
                    loss = (1.0 * criterion(out_train, imgn_train_2) / imgn_train.size()[
                        0] * 2) + (0.024 * criterion(feature_out, feature_img) / (feature_img.size()[0] / 2)) + (
                                1.0 * criterion(out_dn, imgn_train_mid) / imgn_train.size()[
                            0] * 2) + (
                                1.0 * criterion(out_wm, imgn_train_2) / imgn_train.size()[
                            0] * 2) + (0.024 * criterion(feature_out_wm, feature_img) / (feature_img.size()[0] / 2))
                else:
                    loss = (1.0 * criterion(out_train, imgn_train_2) / imgn_train.size()[
                        0] * 2) + (0.024 * criterion(feature_out, feature_img) / (feature_img.size()[0] / 2))
                    
                loss.backward()
                optimizer.step()
                
                # results
                model.eval()
                if opt.net == "FFDNet":
                    out_train = torch.clamp(model(imgn_train, noise_sigma), 0., 1.)
                elif opt.net == "DPUNet" or opt.net == "PSLNet":
                    out_train = torch.clamp(model(imgn_train)[0], 0., 1.)
                elif opt.net == "SSHNet":
                    out_train = torch.clamp(model(imgn_train)[0], 0., 1.)
                else:
                    out_train = torch.clamp(model(imgn_train), 0., 1.)

                psnr_train = batch_PSNR(out_train, img_train, 1.)
                
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                    (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
                
                step += 1
                if step % 10 == 0:
                    writer.add_scalar("PSNR", psnr_train, step)
                    writer.add_scalar("loss", loss.item(), step)
                
                pbar.set_postfix({
                    'Loss': loss.item(),
                    'PSNR_train': psnr_train,
                    'Batch': f'{i + 1}/{len(loader_train)}'
                })

            ## the end of each epoch
            model.eval()
            # Save the trained network parameters
            torch.save(model.state_dict(), os.path.join(opt.outf, model_name))

            # Save into the save folder
            if epoch >= 90:
                torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name[:-4]}_epoch_{epoch + 1}.pth"))
    writer.close()


if __name__ == "__main__":
    # data preprocess
    if opt.preprocess:
        prepare_data(data_path=config['train_data_path'], patch_size=256, stride=128, aug_times=1, mode='color')
    main()
