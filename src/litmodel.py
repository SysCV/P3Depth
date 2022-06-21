"""
Author: Vaishakh Patil
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import dataclasses
from typing import List
import sys
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from src.metrics import evaluate_depth_metrics, DepthMetrics
from src.losses.loss import define_loss, allowed_losses
from src.models import define_model, allowed_models
from src.losses.loss_utils import split_depth2pqrs
from src.utils import save_test_predictions

class DepthLitModel(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.hparams['batch_size'] = config.SOLVER.BATCHSIZE
        self.hparams['learning_rate'] = config.SOLVER.BASE_LR
        self.hparams['epochs'] = config.SOLVER.EPOCH

        if config.LOSS.DEPTH_INIT != '':
            self.depth_loss_init = define_loss(config.LOSS.DEPTH_INIT)

        if config.LOSS.DEPTH_OFFSET != '':
            self.depth_loss_offset = define_loss(config.LOSS.DEPTH_OFFSET)

        if config.LOSS.TYPE != '':
            self.depth_loss_final = define_loss(config.LOSS.TYPE)

        if self.config.LOSS.SSIM:
            self.ssim = define_loss('ssim')

        if config.LOSS.SMOOTH != '':
            self.smooth_loss = define_loss('smooth')

        if self.config.LOSS.PATCH != '':
            self.patch_loss = define_loss(config.LOSS.PATCH, self.config)
            self.patch_loss.to("cuda")

        if self.config.MODEL.PQRS and self.config.LOSS.PQRS != '':
            self.pqrs_loss = define_loss(config.LOSS.PQRS)
            # self.pqrs2pqrs_loss = define_loss(config.LOSS.PQRS)

        if config.MODEL.TYPE in allowed_models():
            self.net = define_model(config.MODEL.TYPE, config)
        else:
            raise ValueError('Given model type is not implemented.')

        def _init_params(m):
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # nn.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        self.net.decoder.apply(_init_params)


    def forward(self, batch):
        image, depth = batch['image'], batch['depth']

        output_dict = self.net(image)

        return output_dict

    def training_step(self, batch, batch_idx):

        image, depth = batch['image'], batch['depth']

        # if self.config.LOSS.ONLY_ORIGINAL_DEPTH_SUPERVISION:
        #     depth_completed = depth

        if 'depth_completed' in batch.keys():
            depth_completed = batch['depth_completed']

            if self.config.LOSS.ONLY_COMPLETED_DEPTH_SUPERVISION:
                depth = batch['depth_completed']
        else:
            raise ValueError('Depth supervision type not implemented.')

        # forward
        output_dict = self.net(image, self.current_epoch) #* self.config.DATA.MAX_DEPTH

        loss = 0.0

        if self.config.LOSS.MULTISCALE != '':
            raise ValueError('MULTISCALE loss is not implemented.')

        if self.config.LOSS.DEPTH_INIT != '':
            loss_depth_init = self.config.LOSS.W_depth_init * self.depth_loss_init(output_dict["depth_init", 1], depth)
            loss += loss_depth_init
            self.log_dict({'loss_depth': loss_depth_init}, prog_bar=True)

        # Check whether specified configuration includes supervision of intermediate PQRS representations.
        if self.config.MODEL.PQRS and self.config.LOSS.PQRS != '':
            loss_pqrs = 0.0
            if self.config.LOSS.PQRS == 'plane_pqrs':
                loss_pqrs += self.pqrs_loss(output_dict, depth, self.config.LOSS.KSI, 1)
            elif self.config.LOSS.PQRS == 'l1_pqrs':
                loss_pqrs += self.pqrs_loss(output_dict, depth, 1)
            else:
                raise ValueError('The specified loss for pqrs is not implemented.')
            loss += loss_pqrs
            self.log_dict({'loss_pqrs': loss_pqrs}, prog_bar=True)

        if self.config.MODEL.OFFSET and self.config.LOSS.DEPTH_OFFSET != '':

            loss_depth_offset_d = self.config.LOSS.W_depth_offset * self.depth_loss_offset(output_dict["depth_offset", 1], depth_completed)

            if self.config.LOSS.SSIM:
                loss_depth_offset_ssim =  self.config.LOSS.W_depth_SSIM * self.ssim(output_dict["depth_offset", 1], depth_completed)
                self.log_dict({'loss_depth_offset_d': loss_depth_offset_d}, prog_bar=True)
                self.log_dict({'loss_depth_offset_ssim': loss_depth_offset_ssim}, prog_bar=True)
                loss_depth_offset = self.config.LOSS.W_depth_offset * (loss_depth_offset_ssim + loss_depth_offset_d)
            else:
                loss_depth_offset = loss_depth_offset_d

            self.log_dict({'loss_depth_offset': loss_depth_offset}, prog_bar=True)
            loss += loss_depth_offset

        ## TODO: if only one depth then no final depth required (not in case of only depth or only PQRS pred)
        if self.config.LOSS.TYPE != '':
            loss_depth_final = self.config.LOSS.W_depth_final *  self.depth_loss_final(output_dict["depth_final", 1], depth)
            self.log_dict({'loss_depth_final': loss_depth_final}, prog_bar=True)
            loss += loss_depth_final

        if self.config.LOSS.PATCH != '':
            loss_patch = self.patch_loss(output_dict["depth_final", 1], depth_completed) * self.config.LOSS.RHO
            self.log_dict({'loss_patch: '+self.config.LOSS.PATCH: loss_patch}, prog_bar=True)
            loss += loss_patch

        self.log_dict({'loss_total': loss}, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image, depth = batch['image'], batch['depth']

        # forward
        output_dict = self.net(image, self.current_epoch)

        if self.config.MODEL.OFFSET:
            depth_output = output_dict["depth_final", 1]
        else:
            depth_output = output_dict["depth_init", 1]

        depth_output = torch.nn.functional.interpolate(depth_output, size=[depth.size(2), depth.size(3)],
                                                 mode='bilinear', align_corners=True)

        # calc metrics
        d_metrics: DepthMetrics = evaluate_depth_metrics(depth_output, depth, self.config.DATASET.TYPE, self.config.DATA.MAX_DEPTH)

        self.log_dict(dataclasses.asdict(d_metrics))

    def test_step(self, batch, batch_idx):

        self.patch_loss = define_loss('patch_approx', self.config)
        # image, depth, path = batch['image'], batch['depth'], batch['path']
        image, depth = batch['image'], batch['depth']

        # forward
        output_dict = self.net(image)

        if self.config.MODEL.OFFSET:
            depth_output = output_dict["depth_final", 1]
        else:
            depth_output = output_dict["depth_init", 1]

        depth_output = torch.nn.functional.interpolate(depth_output, size=[depth.size(2), depth.size(3)], mode='nearest')

        # calc metrics
        d_metrics: DepthMetrics = evaluate_depth_metrics(depth_output, depth, self.config.DATASET.TYPE, self.config.DATA.MAX_DEPTH)

        self.log_dict(dataclasses.asdict(d_metrics))

        if self.config.TEST.SAVE_RESULTS:
            save_test_predictions(self.config, batch, output_dict)

    def configure_optimizers(self):
        config = self.config
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=config.SOLVER.BASE_LR,
                                     weight_decay=config.SOLVER.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            config.SOLVER.LR_STEP_SIZE,
            config.SOLVER.LR_GAMMA,
        )
        return [optimizer], [scheduler]