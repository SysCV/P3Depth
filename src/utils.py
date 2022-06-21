"""
Author: Vaishakh Patil
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import os
import errno
import matplotlib
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Optional, List
from omegaconf import OmegaConf, DictConfig
from src.metrics import recover_metric_depth

def load_config(cfg_path: Optional[str] = None,
                model_cfg_path: Optional[str] = None,
                dataset_cfg_path: Optional[str] = None,
                exp_cfg_path: Optional[str] = None,
                default_cfg_path: str = 'configs/default.yaml',
                update_dotlist: Optional[List[str]] = None) -> DictConfig:

    config = OmegaConf.load(default_cfg_path)

    if cfg_path is not None:
        optional_config = OmegaConf.load(cfg_path)
        config = OmegaConf.merge(config, optional_config)

    if model_cfg_path is not None:
        optional_config = OmegaConf.load(model_cfg_path)
        config = OmegaConf.merge(config, optional_config)

    if dataset_cfg_path is not None:
        optional_config = OmegaConf.load(dataset_cfg_path)
        config = OmegaConf.merge(config, optional_config)

    if exp_cfg_path is not None and exp_cfg_path != '':
        optional_config = OmegaConf.load(exp_cfg_path)
        config = OmegaConf.merge(config, optional_config)

    if update_dotlist is not None:
        update_config = OmegaConf.from_dotlist(update_dotlist)
        config = OmegaConf.merge(config, update_config)

    OmegaConf.set_readonly(config, True)

    return config

def update_config(config: DictConfig = None, update_dotlist: Optional[List[str]] = None) -> DictConfig:

    OmegaConf.set_readonly(config, False)

    if update_dotlist is not None:
        update_config = OmegaConf.from_dotlist(update_dotlist)
        config = OmegaConf.merge(config, update_config)

    OmegaConf.set_readonly(config, True)

    return config


def print_config(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

def check_machine(config: DictConfig, data_dir, out_dir) -> None:
    import socket

    machine_name=socket.gethostname()

    print("Running on :" + machine_name)

    if not ('biwirender' in machine_name or 'bmicgpu' in machine_name):

        if data_dir is None or out_dir is None:
            print("Set Dataset and output Paths to Euler...")
        else:
            print("Changing Dataset and output Paths to Euler...")
            OmegaConf.set_readonly(config, False)
            config.OUTPUT_DIR = out_dir
            config.DATASET.PATH = data_dir
            OmegaConf.set_readonly(config, True)
    return config


def colorize_predictions(value, vmin=None, vmax=None, vmax_95=True, cmap='viridis'):

    value = value.cpu().numpy()[0, 0, :, :]
    # value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    if vmax_95:
        vmax_95 = np.percentile(value, 95)
        vmax = vmax_95 if vmax is None else vmax
    else:
        vmax = value.max() if vmax is None else vmax


    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]
    tensor = torch.from_numpy(img.transpose((2, 0, 1)))/ 255.0

    return tensor


def normalize_predictions(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)

################ Saving test results ###############################

def depth_2_normal(depth, focal_length_x, focal_length_y, lc_window_sz):

    def init_image_coor(height, width):
        x_row = np.arange(0, width)
        x = np.tile(x_row, (height, 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy())  # .cuda()
        u_u0 = x - width / 2.0

        y_col = np.arange(0, height)  # y_col = np.arange(0, height)
        y = np.tile(y_col, (width, 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy())  # .cuda()
        v_v0 = y - height / 2.0
        return u_u0, v_v0

    b, c, h, w = depth.shape
    u_u0, v_v0 = init_image_coor(h, w)
    x = u_u0 * depth / focal_length_x
    y = v_v0 * depth / focal_length_y
    z = depth
    point3D = torch.cat([x, y, z], 1)

    dx = point3D[:, :, :, lc_window_sz:] - point3D[:, :, :, :-lc_window_sz]
    dy = point3D[:, :, :-lc_window_sz,:] - point3D[:, :, lc_window_sz:, :]

    dx = dx[:, :, lc_window_sz:, :]
    dy = dy[:, :, :, :-lc_window_sz]
    assert (dx.size() == dy.size())

    normal = torch.cross(dx, dy, dim=1)
    assert (normal.size() == dx.size())

    normal = F.normalize(normal, dim=1, p=2)
    return -normal

def vis_normal(normal):
    """
    Visualize surface normal. Transfer surface normal value from [-1, 1] to [0, 255]
    @para normal: surface normal, [h, w, 3], numpy.array
    """
    n_img_L2 = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    n_img_norm = normal / (n_img_L2 + 1e-8)
    normal_vis = n_img_norm * 127
    normal_vis += 128
    normal_vis = normal_vis.astype(np.uint8)
    return normal_vis
"""
reference: https://github.com/NVIDIA/flownet2-pytorch/blob/master/utils/flow_utils.py
"""
def compute_color(u, v):
    """
    compute optical flow color map
    :param u: horizontal optical flow
    :param v: vertical optical flow
    :return:
    """

    height, width = u.shape
    img = np.zeros((height, width, 3))

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

    return img

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel

def offset2flow(u, v):
    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)
    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return img

def depth_to_xyzrgb(depth, rgb, focal_length_x, focal_length_y):

    b, c, h, w = depth.shape

    if rgb.size(2) != h or rgb.size(3) != w :
        rgb = torch.nn.functional.interpolate(rgb, size=(h,w), mode='bilinear')

    def init_image_coords(height, width):
        x_row = np.arange(0, width)
        x = np.tile(x_row, (height, 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        u_u0 = x - width / 2.0

        y_col = np.arange(0, height)
        y = np.tile(y_col, (width, 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        v_v0 = y - height / 2.0

        return u_u0, v_v0

    u_u0, v_v0 = init_image_coords(h, w)
    x = u_u0 * depth / focal_length_x
    y = v_v0 * depth / focal_length_y
    z = depth
    pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1)  # [b, h, w, c]
    rgb = rgb.permute(0, 2, 3, 1)

    xyzrgb = torch.cat([pw, rgb], 3)
    xyzrgb = xyzrgb.contiguous().view(b, -1, 6)

    return xyzrgb

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper. Monodepth2
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def create_eval_dirs(EVAL_DIR, config):

    if not os.path.exists(EVAL_DIR):
        try:
            os.makedirs(EVAL_DIR)
            os.makedirs(EVAL_DIR + '/rgb')
            os.makedirs(EVAL_DIR + '/gt')
            os.makedirs(EVAL_DIR + '/pred')
            os.makedirs(EVAL_DIR + '/pred_cmap')
            os.makedirs(EVAL_DIR + '/pred_disp_cmap')
            os.makedirs(EVAL_DIR + '/pred_pointcloud')
            os.makedirs(EVAL_DIR + '/gt_pointcloud')

            if 'PQRS' in config.MODEL.TYPE:
                os.makedirs(EVAL_DIR + '/P_pred')
                os.makedirs(EVAL_DIR + '/Q_pred')
                os.makedirs(EVAL_DIR + '/R_pred')
                os.makedirs(EVAL_DIR + '/S_pred')
                os.makedirs(EVAL_DIR + '/seed_map_pred')
                os.makedirs(EVAL_DIR + '/seed_map_offset_pred')
                os.makedirs(EVAL_DIR + '/offset_pred')
                os.makedirs(EVAL_DIR + '/offset_pred_refined')
                os.makedirs(EVAL_DIR + '/offset_pred_count')
                os.makedirs(EVAL_DIR + '/pred_init_cmap')
                os.makedirs(EVAL_DIR + '/pred_offset_cmap')

            print("Directories created....")

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def save_test_predictions(config, batch, output_dict):

    NORMALIZE_MEAN = config.DATA.NORMALIZE_MEAN
    NORMALIZE_STD = config.DATA.NORMALIZE_STD
    # inv_normalize = Normalize(
    #     mean=[-NORMALIZE_MEAN[0] / NORMALIZE_STD[0], -NORMALIZE_MEAN[1] / NORMALIZE_STD[1], -NORMALIZE_MEAN[2] / NORMALIZE_STD[2]],
    #     std=[1/ NORMALIZE_STD[0], 1/ NORMALIZE_STD[1], 1/ NORMALIZE_STD[2]]
    # )

    inv_normalize = transforms.Normalize(
        mean=[-NORMALIZE_MEAN[0] / NORMALIZE_STD[0], -NORMALIZE_MEAN[1] / NORMALIZE_STD[1], -NORMALIZE_MEAN[2] / NORMALIZE_STD[2]],
        std=[1/ NORMALIZE_STD[0], 1/ NORMALIZE_STD[1], 1/ NORMALIZE_STD[2]]
    )

    # TODO: check this !!!!
    # EVAL_DIR = config.OUTPUT_DIR + '/eval'
    EVAL_DIR = config.EVAL_DIR

    ## Save image
    # unnorm_batch = inv_normalize(batch)
    # unnorm_image = unnorm_batch['image']

    # full_path = batch['path']
    # path = [os.path.basename(full_path[0])]
    # path[0] = path[0].replace("jpg","png")

    ###### UNCOMMENT THIS IS ORIGINAL ###########
    # full_path = batch['path']
    # dir_path = os.path.normpath(full_path[0])
    # out =  dir_path.split(os.sep)
    # path = out[0] + "_" + os.path.basename(full_path[0])
    # path = [path]
    # path[0] = path[0].replace("jpg","png")
    ########################################

    ########################################
    full_path = batch['path']
    dir_path = os.path.normpath(full_path[0])
    out =  dir_path.split(os.sep)

    EVAL_DIR = os.path.join(EVAL_DIR,out[9])
    if not os.path.exists(EVAL_DIR):
        os.makedirs(EVAL_DIR)
        os.makedirs(EVAL_DIR + '/rgb')
        os.makedirs(EVAL_DIR + '/gt')
        os.makedirs(EVAL_DIR + '/pred')
        os.makedirs(EVAL_DIR + '/pred_cmap')
        os.makedirs(EVAL_DIR + '/pred_disp_cmap')
        os.makedirs(EVAL_DIR + '/P_pred')
        os.makedirs(EVAL_DIR + '/Q_pred')
        os.makedirs(EVAL_DIR + '/R_pred')
        os.makedirs(EVAL_DIR + '/S_pred')
        os.makedirs(EVAL_DIR + '/seed_map_pred')
        os.makedirs(EVAL_DIR + '/seed_map_offset_pred')
        os.makedirs(EVAL_DIR + '/offset_pred')
        os.makedirs(EVAL_DIR + '/offset_pred_refined')
        os.makedirs(EVAL_DIR + '/offset_pred_count')
        os.makedirs(EVAL_DIR + '/pred_init_cmap')
        os.makedirs(EVAL_DIR + '/pred_offset_cmap')

    path = out[0] + os.path.basename(full_path[0])
    path = [path]
    path[0] = path[0].replace("jpg","png")
    ########################################

    # print(path[0])

    depth_output = output_dict["depth_final", 1]
    depth = batch['depth']

    depth[depth==0.0] = config.DATA.MIN_DEPTH

    unnorm_image = inv_normalize(batch['image'])
    image_np = unnorm_image.cpu().squeeze().permute(1,2,0).numpy()
    plt.imsave(EVAL_DIR + '/rgb/' + path[0], image_np/image_np.max())

    ## Save GT Depth
    gt_np = depth.cpu().squeeze().squeeze().numpy()
    #plt.imsave(EVAL_DIR + '/gt/' + path[0], np.log10(gt_np), cmap=config.CMAP)
    plt.imsave(EVAL_DIR + '/gt/' + path[0], gt_np, cmap=config.CMAP)

    # ## Save Predicted Depth
    pred_np = depth_output.cpu().squeeze().squeeze().numpy()

    ## mask = gt_np > 0
    ## pred_depth_masked = pred_np[mask]
    ## gt_depth_masked = gt_np[mask]
    ## ratio = np.median(gt_depth_masked) / np.median(pred_depth_masked)
    ## pred_np *= ratio

    pred_np = recover_metric_depth(pred_np, gt_np) ## Recover metric values
    # plt.imsave(EVAL_DIR + '/pred/' + path[0], pred_np)
    # print("Min Depth: "+str(np.min(pred_np))+"  Max Depth: "+str(np.max(pred_np)))
    depth_pred_img = (pred_np * 256).astype(np.uint16)
    depth_pred_pil = Image.fromarray(depth_pred_img)
    depth_pred_pil.save(EVAL_DIR + '/pred/' + path[0])

    plt.imsave(EVAL_DIR + '/pred_cmap/' + path[0], pred_np, cmap="rainbow")  # config.CMAP)

    # ## Save Init Depth
    depth_init = output_dict["depth_init", 1]
    pred_init_np = depth_init.cpu().squeeze().squeeze().numpy()
    plt.imsave(EVAL_DIR + '/pred_init_cmap/' + path[0], pred_init_np, cmap="rainbow")

    # ## Save Offset Depth
    depth_offset = output_dict["depth_offset", 1]
    pred_offset_np = depth_offset.cpu().squeeze().squeeze().numpy()
    plt.imsave(EVAL_DIR + '/pred_offset_cmap/' + path[0], pred_offset_np, cmap="rainbow")

    pred_disp_np = 1/pred_np
    #pred_disp_np, _ = disp_to_depth(pred_disp_np, min_depth=0.01, max_depth=10.0)
    plt.imsave(EVAL_DIR + '/pred_disp_cmap/' + path[0], np.log10(pred_disp_np), cmap="gist_gray") #config.CMAP)

    if config.TEST.SAVE_POINTCLOUDS:
        ## Save GT pointcloud
        ptcloud = depth_to_xyzrgb(depth, (unnorm_image/torch.max(unnorm_image))*255.0, config.DATA.FX_Depth, config.DATA.FY_Depth)
        ptcloud_np = ptcloud.cpu().numpy()
        np.savetxt(EVAL_DIR + '/gt_pointcloud/' + path[0]+ '.txt', ptcloud_np[0])

        ## Save Pred pointcloud
        ptcloud = depth_to_xyzrgb(depth_output, (unnorm_image/torch.max(unnorm_image))*255.0, config.DATA.FX_Depth, config.DATA.FY_Depth)
        ptcloud_np = ptcloud.cpu().numpy()
        np.savetxt(EVAL_DIR + '/pred_pointcloud/' + path[0]+ '.txt', ptcloud_np[0])

    if config.TEST.SAVE_PQRS:
        if config.MODEL.PQRS:
            for param in ['P','Q', 'R', 'S']:
                output = output_dict[param, 1]
                output = torch.nn.functional.interpolate(output, size=[output.size(2), output.size(3)],
                                                               mode='bilinear', align_corners=True)
                pred_np = output.cpu().squeeze().squeeze().numpy()
                plt.imsave(EVAL_DIR + '/' + param + '_pred/' + path[0], pred_np, cmap= config.CMAP)
        else:
            print("Model doesnt generate PQRS")

    if config.TEST.SAVE_OFFSETS_CONFIDENCE:
        if config.MODEL.OFFSET:
            for param in ['seed_map', 'seed_map_offset']:
                output = output_dict[param, 1]
                output = torch.nn.functional.interpolate(output, size=[output.size(2), output.size(3)],
                                                               mode='bilinear', align_corners=True)
                pred_np = output.cpu().squeeze().squeeze().numpy()

                if "seed_map" in param:
                    plt.imsave(EVAL_DIR + '/' + param + '_pred/' + path[0], pred_np, cmap='Spectral')
                else:
                    plt.imsave(EVAL_DIR + '/' + param + '_pred/' + path[0], pred_np, cmap= config.CMAP)


            ## Save colored offsets as optical flow
            offset = output_dict["offset", 1]
            u = offset[0, :, :, 0].cpu().numpy()
            v = offset[0, :, :, 1].cpu().numpy()
            img_flow = offset2flow(u, v)
            plt.imsave(EVAL_DIR + '/offset_pred/' + path[0], np.uint8(img_flow))


            ## Save colored refined offsets as optical flow
            offset_refined = output_dict["offset_refined", 1]
            u = offset_refined[0, :, :, 0].cpu().numpy()
            v = offset_refined[0, :, :, 1].cpu().numpy()
            img_flow = offset2flow(u, v)
            plt.imsave(EVAL_DIR + '/offset_pred_refined/' + path[0], np.uint8(img_flow))

        else:
            print("Model doesnt generate Offsets and confidences")


        # ## Save offset cout at each pixel as image
        # ocoords = output_dict["ocoords", 1]
        # B, H, W, C = ocoords.size()
        # ocoords[..., 0] = ((H - 1) * (ocoords[..., 0] + 1) / 2).int()
        # ocoords[..., 1] = ((W - 1) * (ocoords[..., 1] + 1) / 2).int()
        #
        # offset_count = torch.zeros(B, 1, H, W)
        #
        # for bt in range(0, B):
        #     for i in range(0, H):
        #         for j in range(0, W):
        #             offset_count[bt, 0, ocoords[bt, i, j, 0].long(), ocoords[bt, i, j, 1].long()] += 1
        #
        # offset_count_np = offset_count.cpu().squeeze().squeeze().numpy()
        # plt.imsave(EVAL_DIR + '/offset_pred_count/' + path[0], offset_count_np, cmap=config.CMAP)