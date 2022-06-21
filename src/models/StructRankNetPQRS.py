"""
Author: Vaishakh Patil
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.models.StructRank_utils import resnet, Resnext_torch
from src.models.StructRank_utils.networks import *
from src.plane_param_layers import *

class Decoder(nn.Module):
    def __init__(self, args, inchannels = [256, 512, 1024, 2048], midchannels = [256, 256, 256, 512], upfactors = [2,2,2,2]):
        super(Decoder, self).__init__()

        self.inchannels = inchannels
        self.midchannels = midchannels
        self.upfactors = upfactors

         # Init
        if args.MODEL.PQRS:
            self.outchannels = 4 #  5 if pqrs + confidence
            self.mutliscale_offsets_adjustment = args.MODEL.MULTISCALE_ADJUSTMENT
        else:
            self.outchannels = 1
            self.mutliscale_offsets_adjustment = False

        self.offset_prediction = False

        self.conv = FTB(inchannels=self.inchannels[3], midchannels=self.midchannels[3])
        self.conv1 = nn.Conv2d(in_channels=self.midchannels[3], out_channels=self.midchannels[2], kernel_size=3, padding=1, stride=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=self.upfactors[3], mode='bilinear', align_corners=True)

        self.ffm2_depth = FFM(inchannels=self.inchannels[2], midchannels=self.midchannels[2], outchannels = self.midchannels[2], upfactor=self.upfactors[2])
        self.ffm1_depth = FFM(inchannels=self.inchannels[1], midchannels=self.midchannels[1], outchannels = self.midchannels[1], upfactor=self.upfactors[1])
        self.ffm0_depth = FFM(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels = self.midchannels[0], upfactor=self.upfactors[0])

        if self.mutliscale_offsets_adjustment:
            self.outconv_depth = AO(inchannels=self.inchannels[0] + 4, outchannels=self.outchannels, upfactor=2)
            self.lpg5 = local_planar_guidance(256, 8, args.DATA.MAX_DEPTH)
            self.lpg4 = local_planar_guidance(256, 4, args.DATA.MAX_DEPTH)
            self.lpg3 = local_planar_guidance(256, 2, args.DATA.MAX_DEPTH)
            self.lpg2 = local_planar_guidance(256, 1, args.DATA.MAX_DEPTH)
        else:
            self.outconv_depth = AO(inchannels=self.inchannels[0], outchannels=self.outchannels, upfactor=2)


        # if args.MODEL.PQRS:
        if args.MODEL.OFFSET:

            self.offset_prediction = True
            self.outchannels_o = 4

            self.offset_threshold = args.MODEL.OFFSET_THRESHOLD
            if self.offset_threshold > 1.0 or self.offset_threshold < 0.1:
                raise ValueError('The offset threshold should be between 0.1 - 1.0.')

            self.ffm2_offset = FFM(inchannels=self.inchannels[2], midchannels=self.midchannels[2], outchannels = self.midchannels[2], upfactor=self.upfactors[2])
            self.ffm1_offset = FFM(inchannels=self.inchannels[1], midchannels=self.midchannels[1], outchannels = self.midchannels[1], upfactor=self.upfactors[1])
            self.ffm0_offset = FFM(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels = self.midchannels[0], upfactor=self.upfactors[0])
            self.outconv_offset = AO(inchannels=self.inchannels[0], outchannels=self.outchannels_o, upfactor=2)
            self.tanh = nn.Tanh()
            self.sigmoid = nn.Sigmoid()


    def forward(self, image_features, epoch):

        outputs = {}

        _,_,h,w = image_features[3].size()
        bottleneck = self.conv(image_features[3])
        bottleneck = self.conv1(bottleneck)
        bottleneck = self.upsample(bottleneck)

        ## Depth branch
        outputs["depth_feat", 4] = self.ffm2_depth(image_features[2], bottleneck)
        outputs["depth_feat", 3] = self.ffm1_depth(image_features[1], outputs["depth_feat", 4])
        outputs["depth_feat", 2] = self.ffm0_depth(image_features[0], outputs["depth_feat", 3])

        if self.offset_prediction:

            ## Offset branch
            outputs["offset_feat", 4] = self.ffm2_offset(image_features[2], bottleneck)
            outputs["offset_feat", 3] = self.ffm1_offset(image_features[1], outputs["offset_feat", 4])
            outputs["offset_feat", 2] = self.ffm0_offset(image_features[0], outputs["offset_feat", 3])
            outputs["offset_feat", 1] = self.outconv_offset(outputs["offset_feat", 2])

            outputs["seed_map", 1] = self.sigmoid(outputs["offset_feat", 1][:, 0, :, :]).unsqueeze(1)
            outputs["offset", 1] = self.tanh(outputs["offset_feat", 1][:, 1:3, :, :]) * float(self.offset_threshold)

        if self.mutliscale_offsets_adjustment:
            upsample_size = outputs["depth_feat", 2].size()

            depth_lpg5 = F.interpolate(self.lpg5(bottleneck).unsqueeze(1), (upsample_size[2], upsample_size[3]))
            depth_lpg4 = F.interpolate(self.lpg4(outputs["depth_feat", 4]).unsqueeze(1), (upsample_size[2], upsample_size[3]))
            depth_lpg3 = F.interpolate(self.lpg3(outputs["depth_feat", 3]).unsqueeze(1), (upsample_size[2], upsample_size[3]))
            depth_lpg2 = F.interpolate(self.lpg2(outputs["depth_feat", 2]).unsqueeze(1), (upsample_size[2], upsample_size[3]))

            plane_feat_combined = torch.cat([outputs["depth_feat", 2], depth_lpg2, depth_lpg3, depth_lpg4, depth_lpg5],dim=1)
            outputs["depth_feat", 1] = self.outconv_depth(plane_feat_combined)

            # chunk = self.outconv_depth(plane_feat_combined)
            # outputs["depth_feat", 1] = chunk[: , 0:4, :, :]
            # outputs["conf_depth", 1] = chunk[:, 4, :, :].unsqueeze(1)
        else:
        # if not self.mutliscale_offsets_adjustment:
            outputs["depth_feat", 1] = self.outconv_depth(outputs["depth_feat", 2])

            # chunk = self.outconv_depth(outputs["depth_feat", 2])
            # outputs["depth_feat", 1] = chunk[:, 0:4, :, :]
            # outputs["conf_depth", 1] = chunk[:, 4, :, :].unsqueeze(1)

        return outputs

class StructRankNetPQRS(nn.Module):
    __factory = {
        18: resnet.resnet18,
        34: resnet.resnet34,
        50: resnet.resnet50,
        101: resnet.resnet101,
        152: resnet.resnet152
    }
    def __init__(self, args):

        super(StructRankNetPQRS, self).__init__()
        self.args = args
        self.backbone = args.MODEL.BACKBONE # 'resnet' #'resnext50_32x4d'
        self.pretrained = False # True
        self.inchannels = [256, 512, 1024, 2048]
        self.midchannels = [256, 256, 256, 512]
        self.upfactors = [2, 2, 2, 2]
        self.outchannels = 4

        # Build model
        if 'resnet' in self.backbone:
            self.depth = int(args.MODEL.BACKBONE[6:])
            if self.depth not in StructRankNetPQRS.__factory:
                raise KeyError("Unsupported depth:", self.depth)
            self.encoder = StructRankNetPQRS.__factory[self.depth](pretrained=self.pretrained)
        elif self.backbone == 'resnext50_32x4d':
            self.encoder = Resnext_torch.resnext101_32x8d(pretrained=self.pretrained)
        elif self.backbone == 'resnext101_32x8d':
            self.encoder = Resnext_torch.resnext101_32x8d(pretrained=self.pretrained)
        else:
            self.encoder = Resnext_torch.resnext101(pretrained=self.pretrained)

        self.decoder = Decoder(args, inchannels=self.inchannels, midchannels=self.midchannels, upfactors=self.upfactors)
        self.parameterized_disparity = parameterized_disparity(args.DATA.MAX_DEPTH)
        self.pqrs2depth = pqrs2depth(args.DATA.MAX_DEPTH)
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        ##################################################
        self.get_coords = get_coords
        self.batch_size = args.SOLVER.BATCHSIZE
        self.H, self.W = args.DATA.CENTER_CROP_SIZE[1], args.DATA.CENTER_CROP_SIZE[0]
        coords = self.get_coords(self.batch_size, self.H, self.W, fix_axis=True)
        self.coords = nn.Parameter(coords, requires_grad=False)


    def forward(self, x, epoch=10):

        self.outputs = {}
        
        x = self.encoder(x)
        self.outputs  = self.decoder(x, epoch)

        if self.args.MODEL.PQRS:
            p1, q1, r1, s1, disp1  = self.parameterized_disparity(self.outputs["depth_feat", 1])
            self.outputs["P", 1] = p1
            self.outputs["Q", 1] = q1
            self.outputs["R", 1] = r1
            self.outputs["S", 1] = s1
        else:
            disp1 = self.outputs["depth_feat", 1]

        self.outputs["disp_init", 1] = disp1
        self.outputs["depth_init", 1] = 1/disp1
        self.outputs["depth_final", 1] = 1 / disp1

        if self.args.MODEL.OFFSET:

            # if not self.args.MODEL.PQRS:
            #     raise ValueError('Cannot use offsets without PQRS.')

            batch_size, C, H, W = self.outputs["depth_init", 1].size()

            if self.H != H or self.W != W:
                coords = self.get_coords(batch_size, H, W, fix_axis=True)
                ocoords_orig = nn.Parameter(coords, requires_grad=False)
            else:
                ocoords_orig = self.coords
                if self.batch_size > batch_size:
                    ocoords_orig = self.coords[0:batch_size]

            offset = self.outputs["offset", 1]
            offset = offset.permute(0, 2, 3, 1)
            ocoords = ocoords_orig + offset
            ocoords = torch.clamp(ocoords, min=-1.0, max=1.0)

            self.outputs["offset", 1] = offset
            self.outputs["ocoords", 1] = ocoords

            if int(self.args.MODEL.ITERATIVE_REFINEMENT) > 0 :
                for _ in range(0, int(self.args.MODEL.ITERATIVE_REFINEMENT)):
                    du = offset[:, :, :, 0].unsqueeze(1)
                    dv = offset[:, :, :, 1].unsqueeze(1)
                    du = du + F.grid_sample(du, ocoords, padding_mode="zeros", align_corners=True)
                    dv = dv + F.grid_sample(dv, ocoords, padding_mode="zeros", align_corners=True)
                    # seed_map_offset = F.grid_sample(seed_map, ocoords, padding_mode="zeros", align_corners=True)
                    offset = torch.cat([du, dv], dim=1)
                    offset = offset.permute(0, 2, 3, 1)
                    ocoords = ocoords_orig + offset
                    ocoords = torch.clamp(ocoords, min=-1.0, max=1.0)

            self.outputs["offset_refined", 1] = offset
            self.outputs["ocoords_refined", 1] = ocoords

            self.outputs["seed_map_offset", 1] = F.grid_sample(self.outputs["seed_map", 1], ocoords,
                                                               padding_mode="zeros", align_corners=True)
            if self.args.MODEL.PQRS:
                self.outputs["P_offset", 1] = F.grid_sample(p1, ocoords, padding_mode="border", align_corners=True)
                self.outputs["Q_offset", 1] = F.grid_sample(q1, ocoords, padding_mode="border", align_corners=True)
                self.outputs["R_offset", 1] = F.grid_sample(r1, ocoords, padding_mode="border", align_corners=True)
                self.outputs["S_offset", 1] = F.grid_sample(s1, ocoords, padding_mode="border", align_corners=True)

                disp_offset = self.pqrs2depth(torch.cat([self.outputs["P_offset", 1],
                                                                  self.outputs["Q_offset", 1],
                                                                  self.outputs["R_offset", 1],
                                                                  self.outputs["S_offset", 1]],dim=1))
            else:
                disp_offset = F.grid_sample(disp1, ocoords, padding_mode="border", align_corners=True)

            self.outputs["disp_offset", 1] = disp_offset
            self.outputs["depth_offset", 1] = ( 1 / disp_offset )

            if int(self.args.MODEL.USE_CONFIDENCE) == 0:
                self.outputs["depth_final", 1] = self.outputs["depth_init", 1] + self.outputs["depth_offset", 1]
            else:
                if int(self.args.MODEL.USE_CONFIDENCE) == 1:
                    confidence_map = self.outputs["seed_map", 1]
                elif int(self.args.MODEL.USE_CONFIDENCE) == 2:
                    confidence_map = self.outputs["seed_map_offset", 1]
                else:
                    raise ValueError('The specified confidence is not implemented.')

                self.outputs["depth_final", 1] = (1 - confidence_map) * self.outputs["depth_init", 1] \
                                             + confidence_map * self.outputs["depth_offset", 1]

                #conf_offset, conf_depth = torch.chunk(self.softmax(torch.cat((confidence_map, self.outputs["conf_depth", 1]), 1)), 2, dim=1)
                #self.outputs["depth_final", 1] = conf_depth * self.outputs["depth_init", 1] + conf_offset * self.outputs["depth_offset", 1]

            self.outputs["disp_final", 1] = 1/self.outputs["depth_final", 1]

        return self.outputs

if __name__ == '__main__':
    net = StructRankNetPQRS()
    print(net)
    inputs = torch.ones(4,3,128,128)
    preds = net(inputs)
    print(preds)
