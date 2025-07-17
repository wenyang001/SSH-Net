#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-12-7 20:23
# @Software: PyCharm
import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DPAUNet, HN
from utils import *
import matplotlib.image as matImage
from tqdm import tqdm
import os
from model_ours import SSHNet

path_save = "test5/"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
config = get_config('configs/config_ours.yaml')


parser = argparse.ArgumentParser(description="watermark removal")
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--model", type=str, default='./data/models/',
                    help='load model')
parser.add_argument("--net", type=str, default="IRCNN", help='Network used in test')
parser.add_argument("--test_data", type=str, default='DPAUNet_test', help='The set of tests we created')
parser.add_argument("--test_noiseL", type=float, default=0, help='noise level used on test set')
parser.add_argument("--alpha", type=float, default=0.3, help="The opacity of the watermark")
parser.add_argument("--alphaL", type=float, default=0.3, help="The opacity of the watermark")
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--mode_wm", type=str, default="S", help='with known alpha level (S) or blind training (B)')
parser.add_argument("--loss", type=str, default="L1", help='The loss function used for training')
parser.add_argument("--self_supervised", type=str, default="True", help='T stands for TRUE and F stands for FALSE')
parser.add_argument("--display", type=str, default="True", help='Whether to display an image')
parser.add_argument("--GPU_id", type=str, default="2", help='GPU_id')

opt = parser.parse_args()
opt.model = config['train_model_out_path_DPAUNet']
os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_id
print(opt)
torch.manual_seed(0)

print("net", opt.net, opt.alphaL, opt.test_noiseL, opt.mode, opt.mode_wm, opt.loss, opt.self_supervised)

if opt.self_supervised == "True":
    model_name_3 = "n2n"
else:
    model_name_3 = "n2c"
if opt.mode == "S":
    model_name_4 = "S" + str(int(opt.test_noiseL))
else:
    model_name_4 = "B"
if opt.loss == "L2":
    model_name_5 = "L2"
else:
    model_name_5 = "L1"
if opt.mode_wm == "S":
    model_name_6 = "aS"
else:
    model_name_6 = "aB"
tensorboard_name = opt.net + model_name_3 + model_name_4 + model_name_5 + model_name_6 + "alpha" + str(opt.alpha)

model_name = tensorboard_name + ".pth"
print(model_name)

flag = False
if flag:
    epoch = 97
    model_name = os.path.join(model_name[:-4], f"{model_name[:-4]}_epoch_{epoch + 1}.pth")

def normalize(data):
    return data / 255.


def water_test():
    print('Loading model ...\n')

    if opt.net == "HN":
        net = HN()
    elif opt.net == "PSLNet":
        net = DPAUNet()
    elif opt.net == "SSHNet":
        net = SSHNet()  
    else:
        assert False


    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.model, model_name), map_location='cuda:0'))
    print('Model loaded from %s' % os.path.join(opt.model, model_name))
    model.eval()
    print('Loading data info ...\n')
    data_path = config['train_data_path']
    print(data_path, opt.test_data)
    files_source = glob.glob(os.path.join(data_path, opt.test_data, '*.jpg'))
    print('Start testing...\n')
    files_source.sort()
    print(files_source)
    print(len(files_source))
    all_psnr_source_avg = 0
    all_ssim_source_avg = 0
    all_mse_source_avg = 0

    all_psnr_avg = 0
    all_ssim_avg = 0
    all_mse_avg = 0
    all_lpips_avg = 0

    wm_id = -1
    # for img_index in tqdm(range(12), desc="Processing Images"):
    for img_index in range(12):
        img_index += 1
        psnr_test = 0
        f_index = 0
        ssim_test = 0
        mse_test = 0
        lpips_test = 0
        psnr_source_avg = 0
        ssim_source_avg = 0
        mse_source_avg = 0

        wm_id += 1
        if opt.display == "True":
            print(wm_id)
            if wm_id != 0:
                continue

        # for f in files_source:
        for f in tqdm(files_source, desc=f"Processing source files for image {img_index}"):
            Img = cv2.imread(f)
            Img = normalize(np.float32(Img[:, :, :]))
            Img = np.expand_dims(Img, 0)
            Img = np.transpose(Img, (0, 3, 1, 2))
            _, _, w, h = Img.shape
            w = int(int(w / 32) * 32)
            h = int(int(h / 32) * 32)
            Img = Img[:, :, 0:w, 0:h]
            ISource = torch.Tensor(Img)
            
            # add watermark if alpha is not 0
            if opt.alphaL != 0.0 and opt.alphaL != 0: 
                # print("add watermark")
                INoisy = add_watermark_noise_test(ISource, 0., img_id=img_index, scale_img=1.0, alpha=opt.alphaL)
            else:
                # print("no watermark")
                INoisy = ISource

            # add noise to the image 
            if (opt.test_noiseL == 0) & (opt.mode == 'S'): # noise level is 0
                # print("noise level is 0")
                INoisy = torch.Tensor(INoisy)
            else: # noise level is not 0
                noise_gs = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
                INoisy = torch.Tensor(INoisy) + noise_gs
            
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
            
            with torch.no_grad():
                if opt.net == "FFDNet":
                    noise_sigma = opt.test_noiseL / 255.
                    noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(INoisy.shape[0])]))
                    noise_sigma = Variable(noise_sigma)
                    noise_sigma = noise_sigma.cuda()
                    Out = torch.clamp(model(INoisy, noise_sigma), 0., 1.)
                elif opt.net == "SSHNet" or opt.net == "PSLNet":
                    Out = torch.clamp(model(INoisy)[0], 0., 1.)
                    DN = torch.clamp(model(INoisy)[1], 0., 1.)
                    WM = torch.clamp(model(INoisy)[2], 0., 1.)
                else: # PSNR, SUNet, HN, HN2, DPUNet
                    # print(opt.net)
                    # print("INoisy", INoisy.shape)
                    Out = torch.clamp(model(INoisy)[0], 0., 1.)
                    # Out = torch.unsqueeze(Out, 0)
                INoisy = torch.clamp(INoisy, 0., 1.)

            psnr_source = batch_PSNR(INoisy, ISource, 1.0)
            ssim_source = batch_SSIM(INoisy, ISource, 1.0)
            mse_source = batch_RMSE(INoisy, ISource, 1.0)

            psnr_api = batch_PSNR(Out, ISource, 1.)
            ssim_api = batch_SSIM(Out, ISource, 1.)
            mse_api = batch_RMSE(Out, ISource, 1.)
            
            lpips_flag = False
            if lpips_flag:
                lpips_api = batch_LPIPS(Out, ISource)
            else:
                lpips_api = 0

            if opt.display == "True":
                Out_np = Out.cpu().numpy()
                INoisy_np = INoisy.cpu().numpy()
                # print("Out_np", Out_np.shape)
                pic = Out_np[0]
                r, g, b = pic[0], pic[1], pic[2]
                b = b[None, :, :]
                r = r[None, :, :]
                g = g[None, :, :]
                pic = np.concatenate((b, g, r), axis=0)
                pic = np.transpose(pic, (1, 2, 0))
                print("pic.shape", pic.shape)
                save_path1 = data_path + path_save + "/pic_out" + str(f_index) + opt.net + "_psnr_" + str(psnr_api) + "_ssim_" + str(ssim_api) + ".jpg"
                matImage.imsave(save_path1, pic)
                
                pic = INoisy_np[0] # with noise
                r, g, b = pic[0], pic[1], pic[2]
                b = b[None, :, :]
                r = r[None, :, :]
                g = g[None, :, :]
                pic = np.concatenate((b, g, r), axis=0)
                pic = np.transpose(pic, (1, 2, 0))
                matImage.imsave(data_path + path_save + "/pic_input" + str(f_index) +"_psnr_" + str(psnr_source)+ "_ssim_" + str(ssim_source)+".jpg", pic)
                
                DN_np = DN.cpu().numpy()
                pic = DN_np[0] # with noise
                r, g, b = pic[0], pic[1], pic[2]
                b = b[None, :, :]
                r = r[None, :, :]
                g = g[None, :, :]
                pic = np.concatenate((b, g, r), axis=0)
                pic = np.transpose(pic, (1, 2, 0))
                matImage.imsave(data_path + path_save + "/pic_dm" + str(f_index) +"_psnr_" + str(psnr_source)+ "_ssim_" + str(ssim_source)+".jpg", pic)
                
                WM_np = WM.cpu().numpy()
                pic = WM_np[0] # with noise
                r, g, b = pic[0], pic[1], pic[2]
                b = b[None, :, :]
                r = r[None, :, :]
                g = g[None, :, :]
                pic = np.concatenate((b, g, r), axis=0)
                pic = np.transpose(pic, (1, 2, 0))
                matImage.imsave(data_path + path_save + "/pic_wm" + str(f_index) +"_psnr_" + str(psnr_source)+ "_ssim_" + str(ssim_source)+".jpg", pic)

                # ISource_np = ISource.cpu().numpy()
                # pic = ISource_np[0]
                # r, g, b = pic[0], pic[1], pic[2]
                # b = b[None, :, :]
                # r = r[None, :, :]
                # g = g[None, :, :]
                # pic = np.concatenate((b, g, r), axis=0)
                # pic = np.transpose(pic, (1, 2, 0))
                # matImage.imsave(data_path + path_save + "/pic_source" + str(f_index) + "_psnr_" + str(psnr_source)+ "_ssim_" + str(ssim_source)+".jpg", pic)

                f_index += 1

            psnr_test += psnr_api
            ssim_test += ssim_api
            mse_test += mse_api
            lpips_test += lpips_api

            psnr_source_avg += psnr_source
            ssim_source_avg += ssim_source
            mse_source_avg += mse_source

        psnr_test /= len(files_source)
        ssim_test /= len(files_source)
        mse_test /= len(files_source)
        lpips_test /= len(files_source)

        psnr_source_avg /= len(files_source)
        ssim_source_avg /= len(files_source)
        mse_source_avg /= len(files_source)

        all_psnr_avg += psnr_test
        all_mse_avg += mse_test
        all_ssim_avg += ssim_test
        all_lpips_avg += lpips_test

        all_psnr_source_avg += psnr_source_avg
        all_mse_source_avg += mse_source_avg
        all_ssim_source_avg += ssim_source_avg

    all_ssim_source_avg /= 12
    all_mse_source_avg /= 12
    all_psnr_source_avg /= 12

    all_ssim_avg /= 12
    all_mse_avg /= 12
    all_psnr_avg /= 12
    all_lpips_avg /= 12

    print("\nPSNR on test data %f SSIM on test data %f LPIPS on test data %f" % (all_psnr_avg, all_ssim_avg, all_lpips_avg))


if __name__ == "__main__":
    # main()
    water_test()
