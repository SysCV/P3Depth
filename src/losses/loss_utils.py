"""
Author: Vaishakh Patil
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import kornia
import numpy as np


def gaussian_derivative_filter(I, sigma):
    # Arguments:
    # I         array-like image, e.g. NumPy array or input with cv2.imread
    # sigma     positive scalar, serving as standard deviation for Gaussian derivative filter. Usual range is in the
    #           order of a pixel, i.e., 1.0.

    # Compute the gradient along the horizontal dimension of the image.
    I_x = gaussian_filter1d(I, sigma, order=0, axis=0)
    I_x = gaussian_filter1d(I_x, sigma, order=1, axis=1)

    # Compute the gradient along the vertical dimension of the image.
    I_y = gaussian_filter1d(I, sigma, order=0, axis=1)
    I_y = gaussian_filter1d(I_y, sigma, order=1, axis=0)

    return I_x, I_y


def split_depth2pqrs(depth): # , upratio

    # converting to gray scale
    disp = 1./(depth + 0.01)

    batch_size, H, W = disp.size()[0], disp.size()[2], disp.size()[3]

    U_coord = torch.arange(start=0, end=W).unsqueeze(0).repeat(H, 1).float()
    V_coord = torch.arange(start=0, end=H).unsqueeze(1).repeat(1, W).float()
    coords = torch.stack([U_coord, V_coord], dim=0)
    coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    coords = coords.permute(0, 2, 3, 1).cuda()
    coords[..., 0] /= W - 1
    coords[..., 1] /= H - 1
    coords = (coords - 0.5) * 2
    U_coord = coords[..., 0].unsqueeze(1)
    V_coord = coords[..., 1].unsqueeze(1)

    ## using sobel filter
    disp_blurred = kornia.gaussian_blur2d(disp, (3, 3), (1.5, 1.5))
    grad = kornia.spatial_gradient(disp_blurred, mode='sobel', order=1, normalized=False)
    param_p = grad[ :, :, 0, :, :]
    param_q = grad[ :, :, 1, :, :]


    pu = torch.mul(param_p, U_coord)
    qv = torch.mul(param_q, V_coord)
    param_r = disp - pu - qv

    param_s = torch.sqrt(param_p ** 2 + param_q ** 2 + param_r ** 2)

    norm_param_p = torch.div(param_p, param_s)
    norm_param_q = torch.div(param_q, param_s)
    norm_param_r = torch.div(param_r, param_s)

    return norm_param_p, norm_param_q, norm_param_r, param_s

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out