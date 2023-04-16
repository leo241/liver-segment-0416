import random

import pandas as pd
import numpy as np
import nibabel as nib  # 处理.nii类型图片
# import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
from random import randint

from AgScResUnet import UNet
from PIL.PngImagePlugin import PngImageFile
import warnings
from train_T1 import DataProcessor

warnings.filterwarnings("ignore")  # ignore warnings

random.seed(123)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速


if __name__ == '__main__':
    # hyper parameter
    name = '52_ximenzi_zhangxiaomin_T1.nii'  # 0.476（73）
    # name = '19_lianying_caiyingxiong_T1.nii' # 0.449（67）
    # name = '7_feilipu_chenyongshun_T1.nii' # 0.467(61) 测试集
    print(name)
    save_dir = 'model_save'
    load_model = '760000.pth'
    threshold = 0.5
    standard = False

    for name in ['31_tongyong_caomeiying_T1.nii','52_ximenzi_zhangxiaomin_T1.nii','19_lianying_caiyingxiong_T1.nii','7_feilipu_chenyongshun_T1.nii']:
        for modid in range(610000, 910001, 30000):
            load_model = f'{modid}.pth'

            class_num = 2
            net = UNet(class_num)
            net.load_state_dict(torch.load(f'{save_dir}/{load_model}')) # 在此更改载入的模型
            net = net.to(device)  # 加入gpu
            def predict(net,target,slice_resize = (256,256)):
                '''
                给定模型和图片，以及网络预测所需要的resize，预测mask，返回mask矩阵
                :param net:
                :param target:
                :return:
                '''
                if type(target) == str:
                    img_target = Image.open(target)
                    origin_size = img_target.size
                    img_arr = np.asarray(img_target.resize(slice_resize,0))
                elif type(target) == PngImageFile or type(target) ==Image.Image:
                    origin_size = target.size
                    img_arr = np.asarray(target.resize(slice_resize,0))
                elif type(target) == np.ndarray:
                    origin_size = target.shape
                    img_arr = np.asarray(Image.fromarray(target).resize(slice_resize,0))
                else:
                    print('<target type error>')
                    return False
                TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
                    transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
                ])
                img_tensor = TensorTransform(img_arr)
                img_tensor4d = img_tensor.unsqueeze(0)  # 只有把图像3维（1，256，256）扩展成4维（1，1，256，256）才能放进神经网络预测
                img_tensor4d = img_tensor4d.to(device)

                # print(type(img_tensor4d), net(img_tensor4d))
                return img_tensor4d, net(img_tensor4d)

            # 51
            glist = []
            lblb = sitk.ReadImage(f'D:/study/pga/dataset/mydata2/image/{name}')
            lblb = sitk.GetArrayFromImage(lblb)
            if standard:
                lblb = (lblb - np.mean(lblb)) / np.std(lblb)  # a.先对图像做标准化
                minimum = np.min(lblb)
                gap = np.max(lblb) - minimum
                lblb = (lblb - minimum) / gap * 255  # b.再对图像做0-255“归一化”
            resize_shape = (lblb.shape[2], lblb.shape[1])
            for id in range(lblb.shape[0]):
                img = lblb[id].squeeze().astype(float)
                y_predict_arr=predict(net,img)[1][1].squeeze(0).squeeze(0).cpu().detach().numpy()
                # img1 = y_predict_arr[1, :, :] < y_predict_arr[0, :, :]
                img1 = y_predict_arr[0, :, :] > threshold # 修改此处更改生成label判定方式
                img1 = Image.fromarray(img1).convert('L')
                img_resize = img1.resize(resize_shape, 0)
                img_resize = np.asarray(img_resize)
                img_resize = np.expand_dims(img_resize, 0)
                glist.append(img_resize/255)
            tmp = np.concatenate(glist, 0)
            tmp_simg = sitk.GetImageFromArray(tmp)
            sitk.WriteImage(tmp_simg, f'predict/{name}.gz')
            # resize_shape = (lblb.shape[2], lblb.shape[1])
            # for item in test_list:
            #     img, label = item[0], item[1]
            #     biz_type, person_name, MRI_type = item[3], item[4],item[5]
            #     img_tensor, pre = predict(net, img)
            #     y_predict_arr = pre[0].squeeze(0).squeeze(0).cpu().detach().numpy()
            #     y_true_arr = np.asarray(label)
            #     y_pre_list.append(y_predict_arr)
            #     y_true_list.append(y_true_arr)
            #     if biz_type == bt and person_name == pt and MRI_type == mt:
            #         img1 = y_predict_arr[1, :, :] < y_predict_arr[0, :, :]
            #         img1 = Image.fromarray(img1).convert('L')
            #         img_resize = img1.resize(resize_shape, 0)
            #         img_resize = np.asarray(img_resize)
            #         # img_resize = np.flipud(img_resize)
            #         # img_resize = np.fliplr(img_resize)
            #         img_resize = np.expand_dims(img_resize, 0)
            #         glist.append(img_resize/255)
            #
            # tmp = np.concatenate(glist, 0)
            # tmp_simg = sitk.GetImageFromArray(tmp)
            # sitk.WriteImage(tmp_simg, f'mask_practice.nii')
            # print(sitk.GetArrayFromImage(tmp_simg).shape)
            gt = sitk.GetArrayFromImage(sitk.ReadImage(f'D:/study/pga/dataset/mydata2/label/{name}')) # ground truth
            print(name, modid, DataProcessor().dice_score(gt,sitk.GetArrayFromImage(tmp_simg),1))





