import random
random.seed(123)
import pandas as pd
import numpy as np
import nibabel as nib  # 处理.nii类型图片
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL import ImageEnhance
from PIL.ImageFilter import BLUR,DETAIL
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from random import randint
from AgScResUnet import UNet
from PIL.PngImagePlugin import PngImageFile
import warnings
from losses import TverskyLoss,GDiceLoss
warnings.filterwarnings("ignore")  # ignore warnings


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速


class DataProcessor:
    def __init__(self):
        # self.foreground = 255
        self.background = 0
        self.pixel = 256
        self.slice_resize = (self.pixel, self.pixel)
        # self.split_ratio = (0.7,0.1,0.2) # 训练集，验证集，测试集的比例

    def mask_one_hot(self,
                     img_arr):  # 将label（512，512）转化为标准的mask形式（512，512，class_num）,这里class_num设置为1，所以出来是（l.h.1）而不是（l,h,2）
        img_arr = np.expand_dims(img_arr, 2)  # (256,256)->(256,256,1), 注意这里传进来的不是img，而是label
        mask_shape = img_arr.shape
        mask1 = np.zeros(mask_shape)
        mask2 = np.zeros(mask_shape)
        mask1[img_arr > self.background] = 1  # foreground
        mask2[img_arr == self.background] = 1  # LV
        mask = np.concatenate([mask1, mask2], 2)  # (len,height,class_num = 4)
        # mask = mask1
        return mask

    def get_data(self,augmentation = True):
        train, val, test = 16, 1, 2
        list_dirs = os.listdir('D:/study/pga/dataset/mydata2/image')
        all_files = []
        for dir in list_dirs:
            if 'T1' in dir:
                all_files.append(dir)
        train_list = list()
        val_list = list()
        length = len(all_files)
        print('T1图片总数量：', length)
        print('测试集图片：', all_files[-1])
        print('验证集图片：', all_files[-2])
        transform1 = transforms.CenterCrop(self.slice_resize)
        for file in all_files[0:length - 2]: # 训练集
            img = sitk.ReadImage(f'D:/study/pga/dataset/mydata2/image/{file}')
            img = sitk.GetArrayFromImage(img)
            img = (img - np.mean(img))/np.std(img) # a.先对图像做标准化
            minimum = np.min(img)
            gap = np.max(img) - minimum
            img = (img - minimum) / gap * 255  # b.再对图像做0-255“归一化”
            label = sitk.ReadImage(f'D:/study/pga/dataset/mydata2/label/{file}')
            label = sitk.GetArrayFromImage(label)
            for id in range(img.shape[0]):
                id_prop = id/img.shape[0] # 序号的比例
                random_rotate = randint(1, 360)  # 随机旋转角度
                img1 = img[id, :, :]
                label1 = label[id, :, :]
                img1 = Image.fromarray(img1).convert('L')
                img_resize = img1.resize(self.slice_resize, 0)
                label1 = Image.fromarray(label1).convert('L')
                label_resize = label1.resize(self.slice_resize, 0)
                train_list.append([img_resize, label_resize, id_prop]) # 1.原图像（无增强）
                if augmentation: # 图像增强
                    # train_list.append([transform1(img1), transform1(label1), id_prop]) # 2. 图像增强 - 中心剪裁
                    train_list.append([img_resize.rotate(random_rotate), label_resize.rotate(random_rotate), id_prop]) # 3.图像增强 - 随机角度旋转
                    train_list.append([img_resize.transpose(Image.FLIP_LEFT_RIGHT), label_resize.transpose(Image.FLIP_LEFT_RIGHT), id_prop]) # 4.图像增强 - 左右旋转
                    train_list.append([img_resize.transpose(Image.FLIP_TOP_BOTTOM), label_resize.transpose(Image.FLIP_TOP_BOTTOM), id_prop]) # 5.图像增强 - 上下旋转
                    train_list.append([ImageEnhance.Brightness(img_resize).enhance(factor=1.5), label_resize, id_prop]) # 6.图像增强 - 增加亮度
                    train_list.append([ImageEnhance.Brightness(img_resize).enhance(factor=0.5), label_resize,id_prop])  # 7.图像增强 - 降低亮度
                    train_list.append([ImageEnhance.Color(img_resize).enhance(factor=1.5), label_resize,id_prop])  # 8.图像增强 - 增加色彩饱和
                    train_list.append([ImageEnhance.Color(img_resize).enhance(factor=0.5), label_resize,id_prop])  # 9.图像增强 - 降低色彩饱和
                    train_list.append([ImageEnhance.Contrast(img_resize).enhance(factor=1.5), label_resize,id_prop])  # 10.图像增强 - 增加对比度
                    train_list.append([ImageEnhance.Contrast(img_resize).enhance(factor=0.5), label_resize,id_prop])  # 11.图像增强 - 减低对比度
                    train_list.append([ImageEnhance.Sharpness(img_resize).enhance(factor=1.5), label_resize,id_prop])  # 12.图像增强 - 增加锐化度
                    train_list.append([ImageEnhance.Sharpness(img_resize).enhance(factor=0.5), label_resize,id_prop])  # 13.图像增强 - 减低锐化度
                    train_list.append([img_resize.filter(BLUR), label_resize, id_prop]) # 14.图像增强 - 添加模糊
                    train_list.append([img_resize.filter(DETAIL), label_resize, id_prop])  # 15.图像增强 - 细节增强滤波

        for file in all_files[length - 2:length - 1]: # 验证集
            img = sitk.ReadImage(f'D:/study/pga/dataset/mydata2/image/{file}')
            img = sitk.GetArrayFromImage(img)
            img = (img - np.mean(img)) / np.std(img)  # a.先对图像做标准化
            minimum = np.min(img)
            gap = np.max(img) - minimum
            img = (img - minimum) / gap * 255  # 0，1缩放
            label = sitk.ReadImage(f'D:/study/pga/dataset/mydata2/label/{file}')
            label = sitk.GetArrayFromImage(label)
            for id in range(img.shape[0]):
                img1 = img[id, :, :]
                label1 = label[id, :, :]
                img1 = Image.fromarray(img1).convert('L')
                img_resize = img1.resize(self.slice_resize, 0)
                label1 = Image.fromarray(label1).convert('L')
                label_resize = label1.resize(self.slice_resize, 0)
                val_list.append([img_resize, label_resize, id/img.shape[0]])
        return train_list, val_list

    def dice_score(self, fig1, fig2, class_value):
        '''
        计算某种特定像素级类别的DICE SCORE
        :param fig1:
        :param fig2:
        :param class_value:
        :return:
        '''
        fig1_class = fig1 == class_value
        fig2_class = fig2 == class_value
        A = np.sum(fig1_class)
        B = np.sum(fig2_class)
        AB = np.sum(fig1_class & fig2_class)
        if A + B == 0:
            return 1
        return 2 * AB / (A + B)


class MyDataset(Dataset):  #
    '''
    继承了torch.utils.data.Dataset,用于加载数据，后续载入神经网络中
    '''

    def __init__(self, data, TensorTransform):
        self.data = data
        self.TensorTransform = TensorTransform

    def __getitem__(self, item):  # 这个是Dataset类的关键函数，形成数据的最终形式，通过迭代的形式喂给后续的神经网络
        img, mask, id = self.data[item]
        img_arr = np.asarray(img)
        img_arr = np.expand_dims(img_arr, 2)  # (256,256)->(256，256,1) # 实际图像矩阵
        mask = DataProcessor().mask_one_hot(np.asarray(mask))

        return self.TensorTransform(img_arr), self.TensorTransform(mask), torch.tensor(id)

    def __len__(self):
        return len(self.data)


class nn_processor:
    def __init__(self, train_loader, valid_loader=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train(self, net, lr=0.0001, EPOCH=40, max_iter=500, save_iter=500, print_iter=100, first_iter=0,
              loss_func=nn.MSELoss(), loss_func2=TverskyLoss(),save_dir = 'model_save',roll = False):
        loss3 = nn.BCELoss()
        loss4 = nn.BCEWithLogitsLoss()
        net = net.to(device)  # 加入gpu
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        i = 0
        # loss_train_list = list()
        # loss_valid_list = list()
        # iter_list = list()
        stop = False
        # create_folder('model_save')
        # create_folder('loss')
        for epoch in range(EPOCH):
            if stop == True:
                break
            for step, (img, label, slice_id) in enumerate(self.train_loader):
                # print(torch.mean(img))
                # print(torch.mean(label))
                # print(slice_id)

                img, label, slice_id = img.to(device), label.to(device), slice_id.to(device)
                pid, plabel = net(img)
                # print(output.shape) # (batchsize,classnum,l,h)
                # print(label.shape)       # (batchsize,classnum,l,h)

                # print(tlabelpe(pid),tlabelpe(label),tlabelpe(plabel),tlabelpe(slice_id))
                # print(loss_func(pid, label))
                # print(loss_func2(plabel,slice_id))
                pid = pid.to(torch.float)
                label = label.to(torch.float)
                plabel = plabel.to(torch.float)
                slice_id = slice_id.to(torch.float)
                # loss = loss_func(pid, slice_id).to(torch.float) * 0.0001 + loss3(plabel, label)
                loss = loss3(plabel, label)
                # loss = loss_func2(plabel,slice_id).to(torch.float)
                # print(loss, tlabelpe(loss))
                # print('\n')
                # loss = loss_func(pid, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1

                if i % print_iter == 0:
                    print(f'\n epoch:{epoch + 1}\niteration: {i + first_iter}')
                    if i > max_iter:  # 达到最大迭代，保存模型
                        stop = True
                        torch.save(net.state_dict(), f'{save_dir}/{i + first_iter}.pth')
                        print('\n model saved!')
                        break
                    if i % save_iter == 0:  # 临时保存
                        if roll:
                            try:
                                os.remove(f'{save_dir}/{i + first_iter - save_iter}.pth')  # 日志回滚，只保留最新的模型
                            except:
                                pass
                        torch.save(net.state_dict(), f'{save_dir}/{i + first_iter}.pth')
                        print(f'\n model temp {i + first_iter} saved!')
                    for data in self.valid_loader:
                        x_valid, y_valid, slice_valid = data
                        x_valid, y_valid, slice_valid = x_valid.to(device), y_valid.to(device), slice_valid.to(device)
                        output1, output2 = net(x_valid)
                        valid_loss = loss3(output2, y_valid.to(torch.float))
                        # loss_train_list.append(float(loss))  # 每隔10个iter，记录一下当前train loss
                        # loss_valid_list.append(float(valid_loss))  # 每隔10个iter，记录一下当前valid loss
                        # iter_list.append(i + first_iter)  # 记录当前的迭代次数
                        print('\n train_loss:', float(loss))
                        print('\n -----valid_loss-----:', float(valid_loss))
                        break


if __name__ == '__main__':
    batch_size = 1  # 设置部分超参数
    class_num = 2
    pretrain = 610000 # 是否使用预训练参数,如果为True则选取save_dir路径下轮数最高的模型，或者可以指定路径下存在的具体的整型轮数
    first_iter = 0
    save_dir = 'model_save' # 模型保存文件夹名称
    load_dir = 'model_save' # 预训练路径
    if_augmentation = False # 训练集是否补充图像增强数据
    save_iter = 30000
    lr = 0.0001
    max_iter = float('inf')
    EPOCH = 100000
    if_roll = False

    dp = DataProcessor()
    train_list, valid_list= dp.get_data(augmentation=if_augmentation)  # 获取训练集，验证集，测试集上的数据（暂时以列表的形式）,并再训练集中添加图像增强
    # def check(id):
    #     img, label = train_list[id][0], train_list[id][1]
    #     plt.imshow(Image.blend(img, label, 0.5))
    #     plt.show()
    #     plt.imshow(img)
    #     plt.show()
    #     print(train_list[id])
    # check(132)
    print(len(train_list), len(valid_list))
    TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
        transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
    ])

    train_data = MyDataset(train_list, TensorTransform=TensorTransform)
    valid_data = MyDataset(valid_list, TensorTransform=TensorTransform)  # 从image2tentor
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                              num_workers=0)  # batch_size是从这里的DataLoader传递进去的
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=0)
    net = UNet(class_num)
    if pretrain:
        names = os.listdir(save_dir)
        names2 = []
        for name in names:
            name_cut = name.replace('.pth','')
            try:
                names2.append(int(name_cut))
            except:
                pass
        if type(pretrain) == int and pretrain in names2:
            first_iter = pretrain
        elif type(pretrain) == int and pretrain not in names2:
            print(names)
            raise Exception('指定了预训练模型但不存在')
        else:
            first_iter = max(names2)
        net.load_state_dict(torch.load(f'{load_dir}/{first_iter}.pth'))

    unet_processor = nn_processor(train_loader, valid_loader)
    unet_processor.train(net, EPOCH=EPOCH, max_iter=max_iter, first_iter=first_iter, lr=lr,save_dir=save_dir,save_iter=save_iter,roll=if_roll)

    # def predict(net,target,slice_resize = (512,512)):
    #     '''
    #     给定模型和图片，以及网络预测所需要的resize，预测mask，返回mask矩阵
    #     :param net:
    #     :param target:
    #     :return:
    #     '''
    #     if type(target) == str:
    #         img_target = Image.open(target)
    #         origin_size = img_target.size
    #         img_arr = np.asarray(img_target.resize(slice_resize,0))
    #     elif type(target) == PngImageFile or type(target) ==Image.Image:
    #         origin_size = target.size
    #         img_arr = np.asarray(target.resize(slice_resize,0))
    #     elif type(target) == np.ndarray:
    #         origin_size = target.shape
    #         img_arr = np.asarray(Image.fromarray(target).resize(slice_resize,0))
    #     else:
    #         print('<target type error>')
    #         return False
    #     TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
    #         transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
    #     ])
    #     img_tensor = TensorTransform(img_arr)
    #     img_tensor4d = img_tensor.unsqueeze(0)  # 只有把图像3维（1，256，256）扩展成4维（1，1，256，256）才能放进神经网络预测
    #     img_tensor4d = img_tensor4d.to(device)
    #
    #     # print(type(img_tensor4d), net(img_tensor4d))
    #     return img_tensor4d, net(img_tensor4d)
    #
    # for item in valid_list:
    #     img, label = item[0], item[1]
    #     img_tensor, pre = predict(net, img)
    #     y_predict_arr = pre[0].squeeze(0).squeeze(0).cpu().detach().numpy()
    #     y_true_arr = np.asarray(label)










