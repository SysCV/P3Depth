'''
Reference: Structure-guided Ranking Loss for Single Image Depth Prediction
Original Paper: Monocular Relative Depth Perception with Web Stereo Data Supervision
'''

import torch
import torch.nn as nn
import torch.nn.init as init

class FTB(nn.Module):
    def __init__(self, inchannels, midchannels=512):
        super(FTB, self).__init__()
        self.in1 = inchannels
        self.mid = midchannels

        self.conv1 = nn.Conv2d(in_channels=self.in1, out_channels=self.mid, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_branch = nn.Sequential(nn.ELU(inplace=True),\
                                         nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3, padding=1, stride=1, bias=True),\
                                         nn.BatchNorm2d(num_features=self.mid),\
                                         nn.ELU(inplace=True),\
                                         nn.Conv2d(in_channels=self.mid, out_channels= self.mid, kernel_size=3, padding=1, stride=1, bias=True))
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv_branch(x)
        x = self.elu(x)

        return x


class FFM(nn.Module):
    # Feature fusion module
    def __init__(self, inchannels, midchannels, outchannels, upfactor=2):
        super(FFM, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.ftb1 = FTB(inchannels=self.inchannels, midchannels=self.midchannels)
        self.ftb2 = FTB(inchannels=self.midchannels, midchannels=self.outchannels)

        self.upsample = nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True)

    def forward(self, low_x, high_x):
        x = self.ftb1(low_x)
        if x.size() != high_x.size():
            high_x = torch.nn.functional.interpolate(high_x, (x.size()[2], x.size()[3]))
        x = x + high_x
        x = self.ftb2(x)
        x = self.upsample(x)

        return x

class AO(nn.Module):
    # Adaptive output module
    def __init__(self, inchannels, outchannels, upfactor=2):
        super(AO, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.adapt_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.inchannels, out_channels=self.inchannels // 2, kernel_size=3, padding=1,
                      stride=1, bias=True), \
            nn.BatchNorm2d(num_features=self.inchannels // 2), \
            nn.ELU(inplace=True), \
            nn.Conv2d(in_channels=self.inchannels // 2, out_channels=self.outchannels, kernel_size=3, padding=1,
                      stride=1, bias=True), \
            nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True))

    def forward(self, x):
        x = self.adapt_conv(x)
        return x
