import torch
from torch import nn
# from torch.nn import functional as F
from losses import *

class Res_Conv_Block(nn.Module): # 自定义模块1-残差卷积快
    def __init__(self,in_channel,out_channel):
        super(Res_Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1), # kernalsize = 3, stride = padding = 1,这样保持输入和输出长宽不变，只改变通道（channel，也是rgb channel）维度
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(), # 暂时修改为leakyrelu，后面可以在元学习或者交叉验证的时候仿照nnUnet进行调参
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            # nn.LeakyReLU() # 如果不是resUnet，这里去掉注释，下面forward直接返回f（x）即可
        )
        self.xreshape = nn.Conv2d(in_channel,out_channel,1,1,0) # 使用1 * 1的卷积核reshape输入的通道数，使得resnet结构x + f（x）变得合法
        self.relu = nn.LeakyReLU()
    def forward(self,x):
        x_reshape =self.xreshape(x)
        f_x = self.layer(x)
        return self.relu(f_x + x_reshape)


class DownSample(nn.Module): # 自定义模块2 - 降采样
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
    def forward(self,x):
        return self.layer(x)

class UpSample(nn.Module): # 自定义模块3 - 上采样
    def __init__(self,channel): # 参数为减半前的通道数
        super(UpSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel, channel // 2, 1, 1),  # 通过1 * 1的卷积核改变将通道数更改为之前的一半
            nn.BatchNorm2d(channel // 2),
            nn.LeakyReLU()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear',align_corners=True)
    def forward(self,x): # 上采样的参数要有一个同层的“前辈”，也就是feature map
        # up=F.interpolate(x,scale_factor=2,mode='nearest')
        up = self.upsample(x) # 这里有两种内置的上采样函数可以选择，我选择了nn.Upsample
        out=self.layer(up)
        # return torch.cat((out,feature_map),dim=1)
        return out

class FullyConnectBlock(nn.Module): # 自定义模块2 - 全连接层
    def __init__(self,in_channel, out_channel):
        super(FullyConnectBlock, self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)


class Gating_Signal(nn.Module): # 自定义模块1-残差卷积快
    def __init__(self,in_channel,out_channel):
        super(Gating_Signal, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,1,1,0), # kernalsize = 3, stride = padding = 1,这样保持输入和输出长宽不变，只改变通道（channel，也是rgb channel）维度
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(), # 暂时修改为leakyrelu，后面可以在元学习或者交叉验证的时候仿照nnUnet进行调参
        )
    def forward(self,x):
        return self.layer(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_channel): # 这里的in_channel是x的通道数，g的通道数是x的二倍
        super(AttentionBlock,self).__init__()
        self.in_channel = in_channel
        self.Upsample = UpSample(in_channel * 2)
        self.conv = nn.Conv2d(in_channel, 1,1,1,0)
        self.bn = nn.BatchNorm2d(1)
        self.sg = nn.Sigmoid()
        self.conv2 = nn.Conv2d(1,in_channel,1,1,0)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.relu = nn.LeakyReLU()
    def forward(self,x,g):
        up_g = self.Upsample(g)
        xg_sum = up_g + x
        act_xg = self.relu(xg_sum)
        psi = self.conv(act_xg)
        psi = self.bn(psi)
        sigmoid_xg =self.sg(psi)
        result = self.conv2(sigmoid_xg)
        result_bn = self.bn2(result)
        return result_bn

class UNet(nn.Module): # 自定义主模块-Unet主架构
    def __init__(self,num_classes):
        super(UNet, self).__init__()
        self.conv1=Res_Conv_Block(1,8) # first para means rgb-channel size,for gray figure choose 1
        self.down1=DownSample(8)

        self.conv2=Res_Conv_Block(8,16) # focal tversky论文中 的参数通道数不改变
        self.down2=DownSample(16)

        self.conv3=Res_Conv_Block(16,32)
        self.down3=DownSample(32)

        self.conv4=Res_Conv_Block(32,64) # focal tversky论文中 的参数通道数不改变
        self.down4=DownSample(64)

        self.center = Res_Conv_Block(64,128)

        # 对center进行全连接层，对应2d-slice序号的回归损失函数
        f1, f2, f3 = 32768, 256 ,16 # 在此更改全连接层的维度
        self.full_connect1 = FullyConnectBlock(f1, f2)
        self.full_connect2 = FullyConnectBlock(f2, f3)
        self.full_connect3 = FullyConnectBlock(f3, 1)

        # 进入上采样右侧部分
        self.g1 =Gating_Signal(128,128)
        self.ag1 = AttentionBlock(64)
        self.up1 = UpSample(128)

        self.g2 = Gating_Signal(128, 64)
        self.ag2 = AttentionBlock(32)
        self.up2 = UpSample(64)

        self.g3 = Gating_Signal(64, 32)
        self.ag3 = AttentionBlock(16)
        self.up3 = UpSample(32)

        self.g4 = Gating_Signal(32, 16)
        self.ag4 = AttentionBlock(8)
        self.up4 = UpSample(16)

        self.out = nn.Conv2d(16,num_classes, 1,1,0)

        # 补充
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # Unet左侧降采样
        C1 = self.conv1(x)
        P1 = self.down1(C1)
        C2 = self.conv2(P1)
        P2 = self.down2(C2)
        C3 = self.conv3(P2)
        P3 = self.down3(C3)
        C4 = self.conv4(P3)
        P4 = self.down4(C4)

        # 中间使用全连接层产生中间结果
        C = self.center(P4)
        C_flatten = C.view(C.shape[0], -1)
        id = self.full_connect3(self.full_connect2(self.full_connect1(C_flatten))) # middle output - loss of slice id

        # 右侧soft-attention-gate + 上采样
        G1 = self.g1(C)
        AT1 = self.ag1(C4, G1)
        UP1 = self.up1(C)
        ATUP1 = torch.cat((AT1,UP1),dim=1)

        G2 = self.g2(ATUP1)
        AT2 = self.ag2(C3, G2)
        UP2 = self.up2(UP1)
        ATUP2 = torch.cat((AT2, UP2), dim=1)

        G3 = self.g3(ATUP2)
        AT3 = self.ag3(C2, G3)
        UP3 = self.up3(UP2)
        ATUP3 = torch.cat((AT3, UP3), dim=1)

        G4 = self.g4(ATUP3)
        AT4 = self.ag4(C1, G4)
        UP4 = self.up4(UP3)
        ATUP4 = torch.cat((AT4, UP4), dim = 1)

        OUT = self.out(ATUP4)
        label = self.sigmoid(OUT)
        return id, label

if __name__ == '__main__':
    batch_size = 1
    x=torch.rand((batch_size,1,256,256)) # (batch-size, rgb_channel_size,length,height)
    net=UNet(2) # 做二分类
    pid, plabel = net(x)
    loss_id = nn.MSELoss()
    loss_label = TverskyLoss()
    # loss2 = nn.CrossEntropyLoss()
    gt_label = torch.rand((batch_size,2,256,256))
    gt_id = torch.rand((batch_size, 1))
    print(loss_id(pid,gt_id), loss_label(plabel,gt_label))

