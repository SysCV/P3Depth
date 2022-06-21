"""
Author: Vaishakh Patil
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

EPSILON = 1e-6

def get_coords(batch_size, H, W, fix_axis=False):
    U_coord = torch.arange(start=0, end=W).unsqueeze(0).repeat(H, 1).float()
    V_coord = torch.arange(start=0, end=H).unsqueeze(1).repeat(1, W).float()
    if not fix_axis:
        U_coord = (U_coord - ((W - 1) / 2)) / max(W, H)
        V_coord = (V_coord - ((H - 1) / 2)) / max(W, H)
    coords = torch.stack([U_coord, V_coord], dim=0)
    coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    coords = coords.permute(0, 2, 3, 1).cuda()
    coords[..., 0] /= W - 1
    coords[..., 1] /= H - 1
    coords = (coords - 0.5) * 2
    return coords

def depth2pqrs(depth): # , upratio

    disp = torch.clamp(disp, min=0.001)
    disp = 1./(depth)

    batch_size, H, W = disp.size()[0], disp.size()[2], disp.size()[3]

    coords = get_coords(batch_size, H, W)
    U_coord = coords[..., 0]
    V_coord = coords[..., 1]

    disp_blurred = kornia.gaussian_blur2d(disp, (3, 3), (1.5, 1.5))
    grad = kornia.spatial_gradient(disp_blurred, mode='sobel', order=1, normalized=False)
    param_p = grad[ :, :, 0, :, :]
    param_q = grad[ :, :, 1, :, :]

    pu = torch.mul(param_p, U_coord)
    qv = torch.mul(param_q, V_coord)
    param_r = disp - pu - qv

    param_s = torch.sqrt(param_p ** 2 + param_q ** 2 + param_r ** 2) + EPSILON

    norm_param_p = torch.div(param_p, param_s)
    norm_param_q = torch.div(param_q, param_s)
    norm_param_r = torch.div(param_r, param_s)

    return norm_param_p, norm_param_q, norm_param_r, param_s


class pqrs2depth(nn.Module):
    def __init__(self, max_depth):
        super(pqrs2depth, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.get_coords = get_coords
        self.max_depth = max_depth

    def forward(self, x, upsample_size = None):

        if upsample_size != None:
            x = F.interpolate(x,(upsample_size[2], upsample_size[3]), mode='bilinear')

        p = x[:, 0, :, :]
        q = x[:, 1, :, :]
        r = x[:, 2, :, :]
        s = x[:, 3, :, :]

        batch_size, H, W = p.size()[0], p.size()[1], p.size()[2]

        coords = self.get_coords(batch_size, H, W)
        U_coord = coords[..., 0]
        V_coord = coords[..., 1]

        U_coord = nn.Parameter(U_coord, requires_grad=False)
        V_coord = nn.Parameter(V_coord, requires_grad=False)

        pu = p * U_coord
        qv = q * V_coord

        disp = (pu + qv + r) * s

        # disp = self.relu(disp) + 0.01
        disp = torch.clamp(disp, min=(1/self.max_depth))

        return disp.unsqueeze(1)

class parameterized_disparity(nn.Module):
    def __init__(self, max_depth):
        super(parameterized_disparity, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.max_depth = max_depth
        self.get_coords = get_coords

    def forward(self, x, epoch=0):


        p = x[:, 0, :, :]
        q = x[:, 1, :, :]
        r = x[:, 2, :, :]
        s = x[:, 3, :, :] # * self.max_depth
        #s = x[:, 3, :, :]

        # TODO: refer to dispnetPQRS
        norm_factor = torch.sqrt((p ** 2 + q ** 2 + r ** 2) + EPSILON)
        p = torch.div(p, norm_factor)
        q = torch.div(q, norm_factor)
        r = torch.div(r, norm_factor)
        # s = s * norm_factor

        batch_size, H, W = x.size()[0], x.size()[2], x.size()[3]

        coords = self.get_coords(batch_size, H, W)
        U_coord = coords[..., 0]
        V_coord = coords[..., 1]

        U_coord = nn.Parameter(U_coord, requires_grad=False)
        V_coord = nn.Parameter(V_coord, requires_grad=False)

        pu = p * U_coord
        qv = q * V_coord

        disp = (pu + qv + r) * s
        disp = torch.clamp(disp, min=(1 / self.max_depth))

        return p.unsqueeze(1), q.unsqueeze(1), r.unsqueeze(1), s.unsqueeze(1), disp.unsqueeze(1)


class local_planar_guidance(nn.Module):
    def __init__(self, num_in_filters, upratio, max_depth):
        super(local_planar_guidance, self).__init__()

        self.reduce_feat = torch.nn.Conv2d(num_in_filters, out_channels=4, bias=False, kernel_size=1, stride=1, padding=0)

        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.upratio = float(upratio)
        self.relu = nn.ReLU()
        self.max_depth = max_depth
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat):

        plane_eq = self.reduce_feat(feat)

        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)

        p = plane_eq_expanded[:, 0, :, :]
        q = plane_eq_expanded[:, 1, :, :]
        r = plane_eq_expanded[:, 2, :, :]
        s = plane_eq_expanded[:, 3, :, :]

        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).cuda()
        u = (u - (self.upratio - 1) * 0.5) / self.upratio

        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio)).cuda()
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        norm_factor = torch.sqrt((p ** 2 + q ** 2 + r ** 2) + EPSILON)
        p = torch.div(p, norm_factor)
        q = torch.div(q, norm_factor)
        r = torch.div(r, norm_factor)
        s = s * norm_factor

        disp = (p * u + q * v + r) * s
        disp = torch.clamp(disp, min=(1 / self.max_depth))

        return (1/disp)
