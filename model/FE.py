import torch.nn as nn
import torch
from model.model_component import *

class Bottleneck(nn.Module):
    def __init__(self, dim):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, bias=False,dilation=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True)
        )
    def forward(self, input):
        return self.bottleneck(input)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, hidden_channel, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channel, 1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, hidden_channel, rate1))
        modules.append(ASPPConv(in_channels, hidden_channel, rate2))
        modules.append(ASPPConv(in_channels, hidden_channel, rate3))
        modules.append(ASPPPooling(in_channels, hidden_channel))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * hidden_channel, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class Cloud_Detection_Module(nn.Module):
    def __init__(self, dim = 48):
        super(Cloud_Detection_Module, self).__init__()
        self.cloud_det = nn.Sequential(
            nn.Conv2d(dim, 2*dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(True),
            nn.Conv2d(2*dim, 4*dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(True),
            nn.Conv2d(4*dim, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.cloud_det(input)

class Feature_Extractor(nn.Module):
    def __init__(self):
        super(Feature_Extractor,self).__init__()
        dim = 32
        self.conv_in = nn.Sequential(
            nn.Conv2d(4, dim,kernel_size = 3, stride =1, padding=1,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True)
            )
        self.ReLU = nn.ReLU(True)
        self.cloud_detection = Cloud_Detection_Module(dim)
        self.bottle1 = Bottleneck(dim)
        self.bottle2 = Bottleneck(dim)
        self.bottle3 = Bottleneck(dim)
        self.bottle4 = Bottleneck(dim)
        self.bottle5 = Bottleneck(dim)
        self.bottle6 = Bottleneck(dim)
        self.ASPP = ASPP(in_channels=dim, hidden_channel= 128, atrous_rates=[12,24,36])
        self.aux_conv = nn.Conv2d(dim, 4, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x):

        out = self.conv_in(x)
        out = self.ReLU(self.bottle1(out) + out)
        
        cloud_mask1 = self.cloud_detection(out)
        out = self.ReLU(self.bottle2(out) * cloud_mask1  + out)
        out = self.ReLU(self.bottle3(out) * cloud_mask1  + out)

        cloud_mask2 = self.cloud_detection(out)
        out = self.ReLU(self.bottle4(out) * cloud_mask2  + out)
        out = self.ReLU(self.bottle5(out) * cloud_mask2  + out)

        out = self.ReLU(self.bottle6(out) + out)
        out = self.ASPP(out)
        pred = self.aux_conv(out)

        return cloud_mask2, out, pred