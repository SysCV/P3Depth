"""
Author: Vaishakh Patil
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
from torchvision.utils import make_grid

import wandb
from src.utils import colorize_predictions, offset2flow

from pytorch_lightning import Callback
from pytorch_lightning.callbacks.progress import ProgressBarBase

# Custom Callback
class DepthPredictionLogger(Callback):
    def __init__(self, config, val_samples, num_samples=8):
        super().__init__()
        self.config = config
        self.num_samples = num_samples
        self.val_samples = val_samples
        # self.cmap = cmap

    def on_validation_epoch_end(self, trainer, lit_model):

        val_samples = self.val_samples

        rgb, depth_gt, p_prediction, q_prediction, r_prediction, s_prediction, depth_init_prediction = ([] for i in range(7))
        depth_offset_prediction, depth_final_prediction, disp_final_prediction, seed_map, seed_map_offset = ([] for i in range(5))
        offset_prediction, normal_prediction = ([] for i in range(2))
        offset_refined_prediction, ocoords_refined = ([] for i in range(2))

        for idx in range(0,self.num_samples):

            # Bring the dict of tensors to CPU
            temp_dict = {'image': self.val_samples['image'][idx].to(device=lit_model.device),
                         'depth': self.val_samples['depth'][idx].to(device=lit_model.device)
                         }
            # Get model prediction
            prediction_dict = lit_model(temp_dict)

            depth_init_prediction.append(colorize_predictions(prediction_dict["depth_init", 1]))

            depth_gt.append(colorize_predictions(self.val_samples['depth'][idx], vmax_95=False))
            temp_image = self.val_samples['image'][idx].cpu()[0]
            rgb.append(temp_image/torch.max(temp_image))

            flag_offset = False

            if self.config.MODEL.OFFSET:

                flag_offset = True

                depth_offset_prediction.append(colorize_predictions(prediction_dict["depth_offset", 1]))
                seed_map.append(colorize_predictions(prediction_dict["seed_map", 1], cmap="Spectral"))
                seed_map_offset.append(colorize_predictions(prediction_dict["seed_map_offset", 1],cmap="Spectral"))
                depth_final_prediction.append(colorize_predictions(prediction_dict["depth_final", 1]))
                disp_final_prediction.append(colorize_predictions((prediction_dict["disp_final", 1])))

                ## Save colored offsets as optical flow
                offset = prediction_dict["offset", 1]
                u = offset[0, :, :, 0].cpu().numpy()
                v = offset[0, :, :, 1].cpu().numpy()
                img_flow_np = offset2flow(u, v)
                img_flow = torch.from_numpy(img_flow_np.transpose((2, 0, 1)))/ 255.0
                offset_prediction.append(img_flow)

                # if "offset_refined" in prediction_dict.keys():
                ## Save colored offsets as optical flow
                offset_refined = prediction_dict["offset_refined", 1]
                u = offset_refined[0, :, :, 0].cpu().numpy()
                v = offset_refined[0, :, :, 1].cpu().numpy()
                img_flow_refined_np = offset2flow(u, v)
                img_flow_refined = torch.from_numpy(img_flow_refined_np.transpose((2, 0, 1))) / 255.0
                offset_refined_prediction.append(img_flow_refined)

        # If trainer is in debug mode
        if trainer.logger is not None:

            if flag_offset:
                # Log the images as wandb Image
                trainer.logger.experiment.log({
                    "RGB": [wandb.Image(make_grid(rgb[dt], nrow=1), caption=f"Images "+str(dt)) for dt in range(0, self.num_samples)],
                    "Depth GT": [
                        wandb.Image(make_grid(depth_gt[dt], nrow=1), caption=f"Depth GT" + str(dt)) for
                        dt in
                        range(0, self.num_samples)],
                    "Depth Init": [
                        wandb.Image(make_grid(depth_init_prediction[dt], nrow=1), caption=f"Depth Init" + str(dt)) for
                        dt in
                        range(0, self.num_samples)],
                    "Depth Offset": [wandb.Image(make_grid(depth_offset_prediction[dt], nrow=1), caption=f"Depth Offset " + str(dt)) for dt in
                            range(0, self.num_samples)],
                    "Depth Final": [wandb.Image(make_grid(depth_final_prediction[dt], nrow=1), caption=f"Depth Final" + str(dt)) for dt in
                            range(0, self.num_samples)],
                    "Disparity Final": [
                        wandb.Image(make_grid(disp_final_prediction[dt], nrow=1), caption=f"Disparity Final" + str(dt)) for dt
                        in
                        range(0, self.num_samples)],
                    "Seed Map": [wandb.Image(make_grid(seed_map[dt], nrow=1), caption=f"Seed Map " + str(dt)) for dt in
                            range(0, self.num_samples)],
                    "Seed Map Offset": [wandb.Image(make_grid(seed_map_offset[dt], nrow=1), caption=f"Seed Map Offset" + str(dt)) for dt in
                                 range(0, self.num_samples)],
                    "Offsets": [
                        wandb.Image(make_grid(offset_prediction[dt], nrow=1), caption=f"Offsets" + str(dt)) for dt in
                        range(0, self.num_samples)],
                    "Offsets refined": [
                        wandb.Image(make_grid(offset_refined_prediction[dt], nrow=1), caption=f"Offsets refined" + str(dt))
                        for dt in range(0, self.num_samples)]

                }, commit=False)

            else:

                # Log the images as wandb Image
                trainer.logger.experiment.log({
                    "RGB": [wandb.Image(make_grid(rgb[dt], nrow=1), caption=f"Images " + str(dt)) for dt in
                            range(0, self.num_samples)],
                    "Depth GT": [
                        wandb.Image(make_grid(depth_gt[dt], nrow=1), caption=f"Depth GT" + str(dt)) for
                        dt in
                        range(0, self.num_samples)],
                    "Depth Init": [
                        wandb.Image(make_grid(depth_init_prediction[dt], nrow=1), caption=f"Depth Init" + str(dt)) for
                        dt in
                        range(0, self.num_samples)]
                }, commit=False)
