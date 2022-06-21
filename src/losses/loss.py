"""
Author: Vaishakh Patil
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.plane_param_layers import depth2pqrs

import numpy as np

EPSILON = 1e-6
MIN_DEPTH = 1e-3


def allowed_losses():
    return loss_dict.keys()


def define_loss(loss_name, *args):
    if loss_name not in allowed_losses():
        raise NotImplementedError('Loss functions {} is not yet implemented'.format(loss_name))
    else:
        return loss_dict[loss_name](*args)


class MAE_loss(nn.Module):
    def __init__(self):
        super(MAE_loss, self).__init__()

    def forward(self, prediction, gt):
        # prediction = prediction[:, 0:1]
        abs_err = torch.abs(prediction - gt)
        mask = (gt > MIN_DEPTH).detach()
        mae_loss = torch.mean(abs_err[mask])
        return mae_loss


class MSE_loss(nn.Module):
    def __init__(self):
        super(MSE_loss, self).__init__()

    def forward(self, prediction, gt):
        #err = prediction[:,0:1] - gt
        err = prediction - gt
        mask = (gt > MIN_DEPTH).detach()
        mse_loss = torch.mean((err[mask])**2)
        return mse_loss


class MAE_log_loss(nn.Module):
    def __init__(self):
        super(MAE_log_loss, self).__init__()

    def forward(self, prediction, gt):
        prediction = torch.clamp(prediction, min=1e-6)
        abs_err = torch.abs(torch.log(prediction+1e-6) - torch.log(gt+1e-6))
        mask = (gt > MIN_DEPTH).detach()
        mae_log_loss = torch.mean(abs_err[mask])
        return mae_log_loss


class MSE_log_loss(nn.Module):
    def __init__(self):
        super(MSE_log_loss, self).__init__()

    def forward(self, prediction, gt):
        mask = (gt > MIN_DEPTH).detach()
        prediction = torch.clamp(prediction, min=MIN_DEPTH)
        err = torch.log(prediction[mask]) - torch.log(gt[mask])
        mae_log_loss = torch.mean(err**2)
        return mae_log_loss


class Silog_loss_variance(nn.Module):
    def __init__(self, variance_focus=0.85):
        super(Silog_loss_variance, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt):
        mask = (depth_gt > MIN_DEPTH).detach()
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class SILogLoss(nn.Module):
#class ScaleInvariantLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()

    def forward(self, prediction, gt):

        mask = (gt > MIN_DEPTH).detach()
        prediction = torch.clamp(prediction, min=1e-6)
        err = torch.log(prediction[mask]) - torch.log(gt[mask])

        # the lambda parameter is set to 0.5
        loss =  torch.sqrt(torch.mean(err ** 2) - (0.85 / ((torch.numel(err)) ** 2)) * (torch.sum(err) ** 2)) * 10.0

        return loss


class Huber_loss(nn.Module):
    def __init__(self, delta=10):
        super(Huber_loss, self).__init__()
        self.delta = delta

    def forward(self, outputs, gt, input, epoch=0):
        outputs = outputs[:, 0:1, :, :]
        err = torch.abs(outputs - gt)
        mask = (gt > MIN_DEPTH).detach()
        err = err[mask]
        squared_err = 0.5*err**2
        linear_err = err - 0.5*self.delta
        return torch.mean(torch.where(err < self.delta, squared_err, linear_err))


class Berhu_loss(nn.Module):
    def __init__(self, delta=0.05):
        super(Berhu_loss, self).__init__()
        self.delta = delta

    def forward(self, prediction, gt, epoch=0):
        # prediction = prediction[:, 0:1]
        err = torch.abs(prediction - gt)
        mask = (gt > MIN_DEPTH).detach()
        err = torch.abs(err[mask])
        c = self.delta*err.max().item()
        squared_err = (err**2+c**2)/(2*c)
        linear_err = err
        return torch.mean(torch.where(err > c, squared_err, linear_err))


class Huber_delta1_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, gt, input):
        mask = (gt > MIN_DEPTH).detach().float()
        loss = F.smooth_l1_loss(prediction*mask, gt*mask, reduction='none')
        return torch.mean(loss)


class Disparity_Loss(nn.Module):
    def __init__(self, order=2):
        super(Disparity_Loss, self).__init__()
        self.order = order

    def forward(self, prediction, gt):
        mask = (gt > MIN_DEPTH).detach()
        gt = gt[mask]
        gt = 1./gt
        prediction = prediction[mask]
        err = torch.abs(prediction - gt)
        err = torch.mean(err**self.order)
        return err


class MSE_loss_uncertainty(nn.Module):
    def __init__(self):
        super(MSE_loss_uncertainty, self).__init__()

    def forward(self, prediction, gt, epoch=0):
        mask = (gt > MIN_DEPTH).detach()
        depth = prediction[:, 0:1, :, :]
        conf = torch.abs(prediction[:, 1:, :, :])
        err = depth - gt
        conf_loss = torch.mean(0.5*(err[mask]**2)*torch.exp(-conf[mask]) + 0.5*conf[mask])
        return conf_loss


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, pred_depth, image):
        # Normalize the depth with mean
        depth_mean = pred_depth.mean(2, True).mean(3, True)
        pred_depth_normalized = pred_depth / (depth_mean + 1e-7)

        # Compute the gradient of depth
        grad_depth_x = torch.abs(pred_depth_normalized[:, :, :, :-1] - pred_depth_normalized[:, :, :, 1:])
        grad_depth_y = torch.abs(pred_depth_normalized[:, :, :-1, :] - pred_depth_normalized[:, :, 1:, :])

        # Compute the gradient of the image
        grad_image_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
        grad_image_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)

        grad_depth_x *= torch.exp(-grad_image_x)
        grad_depth_y *= torch.exp(-grad_image_y)

        return grad_depth_x.mean() + grad_depth_y.mean()



class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    # from monodepth2
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1).mean()


class DirectionAndScalePQRSLoss(nn.Module):
    def __init__(self):
        super(DirectionAndScalePQRSLoss, self).__init__()

    def forward(self, prediction_dict, gt, ksi, scale):
        mask = (gt > 0).detach()

        output_size = prediction_dict["P", scale].size()
        depth_gt_scaled = torch.nn.functional.interpolate(gt, size=(output_size[2], output_size[3]), mode='nearest')
        #p_gt, q_gt, r_gt, s_gt = split_depth2pqrs(depth_gt_scaled)
        p_gt, q_gt, r_gt, s_gt = depth2pqrs(depth_gt_scaled)

        p_prediction = prediction_dict["P", scale]
        q_prediction = prediction_dict["Q", scale]
        r_prediction = prediction_dict["R", scale]
        s_prediction = prediction_dict["S", scale]

        # Direction error - ranges from 0 (prediction is identical to GT) to pi (prediction is opposite from GT).
        # Problems with NaN: https://discuss.pytorch.org/t/nan-gradient-for-torch-cos-torch-acos/9617/2
        #direction_error = torch.acos(p_prediction * p_gt + q_prediction * q_gt + r_prediction * r_gt)
        inner_product = p_prediction * p_gt + q_prediction * q_gt + r_prediction * r_gt
        inner_product = torch.clamp(inner_product, min=-0.99, max=0.99)
        direction_error = torch.acos(inner_product)

        # Normal can be computed for eg. nx = -1 * normalized(p), as per literature
        # https://ieeexplore.ieee.org/document/7900061
        # nx_gt, ny_gt, nz_gt = -1*p_gt, -1*q_gt, -1*r_gt
        # nx_prediction, ny_prediction, nz_prediction = -1 * p_prediction, -1 * q_prediction, -1 * r_prediction
        # normal_dot_product = (nx_prediction * nx_gt + ny_prediction * ny_gt + nz_prediction * nz_gt)

        # direction_error = 1.0 - (p_prediction * p_gt + q_prediction * q_gt + r_prediction * r_gt)

        # Scale error - can grow unbounded with current implementation.
        scale_error = torch.maximum(torch.div(s_prediction, s_gt), torch.div(s_gt, s_prediction)) - 1

        direction_loss = torch.mean(direction_error[mask])
        scale_loss = torch.mean(scale_error[mask])

        return direction_loss + ksi * scale_loss


class L1PQRSLoss(nn.Module):
    def __init__(self):
        super(L1PQRSLoss, self).__init__()

    def forward(self, prediction_dict, gt, scale, pqrs2pqrs_loss=False):

        depth_prediction = prediction_dict["depth", scale]
        depth_gt_scaled = torch.nn.functional.interpolate(gt,
                                                          size=(
                                                              depth_prediction.size()[2], depth_prediction.size()[3]),
                                                          mode='nearest')
        mask = (depth_gt_scaled > 0.1).detach()
        p_gt, q_gt, r_gt, s_gt = depth2pqrs(depth_gt_scaled)

        p_prediction = prediction_dict["P", scale]
        q_prediction = prediction_dict["Q", scale]
        r_prediction = prediction_dict["R", scale]
        s_prediction = prediction_dict["S", scale]

        if pqrs2pqrs_loss:

            po_prediction = prediction_dict["P_offset", scale]
            qo_prediction = prediction_dict["Q_offset", scale]
            ro_prediction = prediction_dict["R_offset", scale]
            so_prediction = prediction_dict["S_offset", scale]

            p_error = torch.abs(po_prediction[mask] - p_gt[mask])
            q_error = torch.abs(qo_prediction[mask] - q_gt[mask])
            r_error = torch.abs(ro_prediction[mask] - r_gt[mask])
            s_error = torch.abs(so_prediction[mask] - s_gt[mask])

        else:

            p_error = torch.abs(p_prediction[mask] - p_gt[mask])
            q_error = torch.abs(q_prediction[mask] - q_gt[mask])
            r_error = torch.abs(r_prediction[mask] - r_gt[mask])
            s_error = torch.abs(s_prediction[mask] - s_gt[mask])

        p_loss = torch.mean(p_error)
        q_loss = torch.mean(q_error)
        r_loss = torch.mean(r_error)
        s_loss = torch.mean(s_error)

        l1_loss = p_loss + q_loss + r_loss + s_loss

        return l1_loss


class PatchPlaneApproxLoss(nn.Module):
    def __init__(self, config):
        super(PatchPlaneApproxLoss, self).__init__()

        fx_d = config.DATA.FX_Depth
        fy_d = config.DATA.FY_Depth
        cx_d = config.DATA.CX_Depth
        cy_d = config.DATA.CY_Depth

        self.focal_length_x = fx_d
        self.focal_length_y = fy_d

        K_np = np.array([[fx_d, 0, cx_d, 0],
                         [0, fy_d, cy_d, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float32)

        self.config = config
        self.batch_size = self.config.SOLVER.BATCHSIZE
        self.height = self.config.DATA.TRAIN_CROP_SIZE[1]
        self.width = self.config.DATA.TRAIN_CROP_SIZE[0]
        self.patch_size = self.config.LOSS.PATCH_SIZE

        inv_K_np = np.linalg.pinv(K_np)
        self.inv_K = torch.from_numpy(inv_K_np).unsqueeze(0).cuda()

        self.u_u0, self.v_v0 = self.init_image_coor()

    def init_image_coor(self):

        x_row = np.arange(0, self.width)
        x = np.tile(x_row, (self.height, 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        u_u0 = x - self.width / 2.0

        y_col = np.arange(0, self.height)
        y = np.tile(y_col, (self.width, 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        v_v0 = y - self.height / 2.0

        return u_u0, v_v0

    def depth_to_xyz(self, depth):
        x = self.u_u0 * depth / self.focal_length_x
        y = self.v_v0 * depth / self.focal_length_y
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1)  # [b, h, w, c]
        return pw

    def get_surface_normal(self, xyz):
        """
        Reference: Comparison of Surface Normal Estimation Methods for Range Sensing Applications
                   and Enforcing geometric constraints of virtual normal for depth prediction.
        """

        x, y, z = torch.unbind(xyz, dim=3)
        x = torch.unsqueeze(x, 0)
        y = torch.unsqueeze(y, 0)
        z = torch.unsqueeze(z, 0)

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z

        patch_weight = torch.ones((1, 1, self.patch_size, self.patch_size), requires_grad=False).cuda()

        xx_patch = nn.functional.conv2d(xx, weight=patch_weight, padding=int(self.patch_size / 2))
        yy_patch = nn.functional.conv2d(yy, weight=patch_weight, padding=int(self.patch_size / 2))
        zz_patch = nn.functional.conv2d(zz, weight=patch_weight, padding=int(self.patch_size / 2))
        xy_patch = nn.functional.conv2d(xy, weight=patch_weight, padding=int(self.patch_size / 2))
        xz_patch = nn.functional.conv2d(xz, weight=patch_weight, padding=int(self.patch_size / 2))
        yz_patch = nn.functional.conv2d(yz, weight=patch_weight, padding=int(self.patch_size / 2))

        ATA = torch.stack([xx_patch, xy_patch, xz_patch, xy_patch, yy_patch, yz_patch, xz_patch, yz_patch, zz_patch],
                          dim=4)
        ATA = torch.squeeze(ATA)
        ATA = torch.reshape(ATA, (ATA.size(0), ATA.size(1), 3, 3))

        eps_identity = 1e-6 * torch.eye(3, device=ATA.device, dtype=ATA.dtype)[None, None, :, :].repeat(
            [ATA.size(0), ATA.size(1), 1, 1])

        ATA = ATA + eps_identity

        x_patch = nn.functional.conv2d(x, weight=patch_weight, padding=int(self.patch_size / 2))
        y_patch = nn.functional.conv2d(y, weight=patch_weight, padding=int(self.patch_size / 2))
        z_patch = nn.functional.conv2d(z, weight=patch_weight, padding=int(self.patch_size / 2))

        AT1 = torch.stack([x_patch, y_patch, z_patch], dim=4)
        AT1 = torch.squeeze(AT1)
        AT1 = torch.unsqueeze(AT1, 3)

        patch_num = 4
        patch_x = int(AT1.size(1) / patch_num)
        patch_y = int(AT1.size(0) / patch_num)
        n_img = torch.randn(AT1.shape).cuda()
        overlap = 0 #self.patch_size // 2 + 1
        for x in range(int(patch_num)):
            for y in range(int(patch_num)):
                left_flg = 0 if x == 0 else 1
                right_flg = 0 if x == patch_num - 1 else 1
                top_flg = 0 if y == 0 else 1
                btm_flg = 0 if y == patch_num - 1 else 1

                at1 = AT1[y * patch_y - top_flg * overlap:(y + 1) * patch_y + btm_flg * overlap,
                      x * patch_x - left_flg * overlap:(x + 1) * patch_x + right_flg * overlap]
                ata = ATA[y * patch_y - top_flg * overlap:(y + 1) * patch_y + btm_flg * overlap,
                      x * patch_x - left_flg * overlap:(x + 1) * patch_x + right_flg * overlap]

                try:
                    n_img_tmp, _ = torch.solve(at1, ata)

                    n_img_tmp_select = n_img_tmp[top_flg * overlap:patch_y + top_flg * overlap,
                                       left_flg * overlap:patch_x + left_flg * overlap, :, :]
                    n_img[y * patch_y:y * patch_y + patch_y, x * patch_x:x * patch_x + patch_x, :, :] = n_img_tmp_select
                except:
                    n_img[y * patch_y:y * patch_y + patch_y, x * patch_x:x * patch_x + patch_x, :, :] = 0.0001

        n_img_L2 = torch.sqrt(torch.sum(n_img ** 2, dim=2, keepdim=True))
        n_img_norm = n_img / n_img_L2

        n_img_norm = n_img_norm[: -1, :-1, ...]

        # re-orient normals consistently
        orient_mask = torch.sum(torch.squeeze(n_img_norm) * torch.squeeze(xyz), dim=2) > 0
        n_img_norm[orient_mask] *= -1
        return n_img_norm

    def surface_normal_from_depth(self, depth, valid_mask=None):
        b, c, h, w = depth.shape
        depth_filter = nn.functional.avg_pool2d(depth, kernel_size=3, stride=1, padding=1)
        xyz = self.depth_to_xyz(depth)
        sn_batch = []
        for i in range(b):
            xyz_i = xyz[i, :][None, :, :, :]
            normal = self.get_surface_normal(xyz_i)
            sn_batch.append(normal)
        sn_batch = torch.cat(sn_batch, dim=3).permute((3, 2, 0, 1))
        mask_invalid = (~valid_mask).repeat(1, 3, 1, 1)
        sn_batch[mask_invalid] = 0.0
        return sn_batch


    def forward(self, prediction, gt, inv_K=None):
        fit_prediction = self.surface_normal_from_depth(prediction, prediction>0)
        fit_gt = self.surface_normal_from_depth(gt, gt>0)
        abs_err = torch.abs(fit_prediction - fit_gt)
        loss = torch.mean(abs_err)
        return loss

loss_dict = {
    'mse': MSE_loss,
    'mae': MAE_loss,
    'log_mse': MSE_log_loss,
    'log_mae': MAE_log_loss,
    'silog_loss_variance': Silog_loss_variance,
    'silog': SILogLoss,
    'huber': Huber_loss,
    'huber1': Huber_delta1_loss,
    'berhu': Berhu_loss,
    'disp': Disparity_Loss,
    'smooth': SmoothnessLoss,
    'uncert': MSE_loss_uncertainty,
    'plane_pqrs': DirectionAndScalePQRSLoss,
    'l1_pqrs': L1PQRSLoss,
    'patch_approx': PatchPlaneApproxLoss,
    'ssim': SSIM
    }

