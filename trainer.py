"""
Author: Vaishakh Patil
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import argparse
import os
from datetime import datetime
import sys
import random
import numpy as np

import torch
from omegaconf import OmegaConf, DictConfig, open_dict
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data import define_dataset
from src.litmodel import DepthLitModel
from src.utils import load_config, print_config,update_config, check_machine, create_eval_dirs
from src.callback import DepthPredictionLogger

WANDB_PJ_NAME = 'p3depth'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a predictor')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional config path. `configs/default.yaml` is loaded by default.')
    parser.add_argument('--model_config', type=str, default=None)
    parser.add_argument('--dataset_config', type=str, default=None)
    parser.add_argument('--exp_config', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='the checkpoint file to resume from')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu-ids', type=int, default=None, nargs='+')
    group.add_argument('--n_gpu', type=int, default=None)
    parser.add_argument("--amp", default=None, help="amp opt level", choices=['O1', 'O2', 'O3'])
    parser.add_argument("--profiler", default=None, help="'simple' or 'advanced'", choices=['simple', 'advanced'])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test", action="store_true")
    # parser.add_argument("--test_path",  type=str, default=None, help='test checkpoint path.')
    parser.add_argument("--data_dir", type=str, default=None, help='data path euler.')
    parser.add_argument("--out_dir", type=str, default=None, help='output path euler.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Overwrite configs. (ex. OUTPUT_DIR=results, SOLVER.NUM_WORKERS=8)')
    return parser.parse_args()

def get_gpus(args: argparse.Namespace):
    if args.gpu_ids is not None:
        gpus = args.gpu_ids
    elif args.n_gpu is not None:
        gpus = args.n_gpu
    else:
        gpus = 1
    gpus = gpus if torch.cuda.is_available() else None
    return gpus


def get_trainer(args: argparse.Namespace, config: DictConfig, dataloader) -> Trainer:

    # amp
    precision = 16 if args.amp is not None else 32

    WANDB_PJ_NAME = config.DATASET.TYPE if config.DATASET.TYPE != '' else WANDB_PJ_NAME

    # logger
    if not args.debug:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        wandb_logger = WandbLogger(project=WANDB_PJ_NAME, save_dir=config.OUTPUT_DIR, name=config.EXP_NAME)
        wandb_logger.log_hyperparams(OmegaConf.to_container(config))
    else:
        print("Running in DEBUG Mode...")
        # print("Dataloader samples reduced to 500...")
        wandb_logger = False

    # checkpoint
    checkpoint_callback = ModelCheckpoint(filename='{epoch:03d}-{rmse:.3f}-{delta1:.3f}',
                                    save_top_k=1, monitor='delta1', mode='max')

    # Samples required by the custom DepthPredictionLogger callback to log predictions.
    num_samples = 10

    val_samples = {}
    for idx in range(0, num_samples):
        dict = next(iter(dataloader.val_dataloader(shuffle=True)))
        for key, value in dict.items():
            if idx == 0:
                val_samples[key] = [dict[key]]
            else:
                val_samples[key].append(dict[key])

    return Trainer(
        max_epochs=config.SOLVER.EPOCH,
        callbacks=[checkpoint_callback, DepthPredictionLogger(config, val_samples, num_samples)],
        resume_from_checkpoint=args.resume,
        default_root_dir=config.OUTPUT_DIR,
        gpus= get_gpus(args),
        profiler=args.profiler,
        logger=wandb_logger,
        precision=precision,
        amp_level=args.amp,
        auto_select_gpus=True,
        # gradient_clip_val=1.0,
        #auto_scale_batch_size='binsearch',
        #progress_bar_refresh_rate=2
        #fast_dev_run=args.debug
    ), wandb_logger


def main():
    args = parse_args()

    # config
    config: DictConfig = load_config(args.config, args.model_config, args.dataset_config, args.exp_config, update_dotlist=args.opts)

    # Change paths based on machine
    config: DictConfig =  check_machine( config, args.data_dir, args.out_dir)

    # modules
    LitDataModule = define_dataset(config.DATASET.TYPE, config)

    ## REPRODUCIBILITY
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    # torch.use_deterministic_algorithms(True)

    if args.test:

        MAIN_EVAL_DIR, _ = os.path.split(config.CKPT_PATH)
        EVAL_DIR = os.path.join(MAIN_EVAL_DIR, 'eval')
        config: DictConfig = update_config(config, ["EVAL_DIR=" + EVAL_DIR])

        model = DepthLitModel.load_from_checkpoint(config.CKPT_PATH, config=config)
        trainer = Trainer(gpus=get_gpus(args), default_root_dir=config.OUTPUT_DIR)

        print('Saving result to folder..'+ EVAL_DIR)
        create_eval_dirs(EVAL_DIR, config)

        print_config(config)
        trainer.test(model, test_dataloaders=LitDataModule.val_dataloader(eval=True))

    else:

        ## Create experiment dir with timestamp
        TIMESTAMP = datetime.now().strftime('%d_%m_%Y-%H%M%S')

        CFG_NAME=''
        if args.exp_config is not None:
            head, tail = os.path.split(args.exp_config)
            CFG_NAME = tail[:-5]

        EXP_NAME = config.MODEL.TYPE + "_" + config.MODEL.BACKBONE + "_" + TIMESTAMP + "_" + CFG_NAME
        OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, EXP_NAME+ "_" + config.DATASET.TYPE)
        config: DictConfig = update_config(config, ["EXP_NAME=" + EXP_NAME, "OUTPUT_DIR=" + OUTPUT_DIR])
        print_config(config)

        trainer, wandb_logger = get_trainer(args, config, LitDataModule)
        model = DepthLitModel(config)

        if config.RESUME != '':
            print("=> using pre-trained model '{}'".format(config.RESUME))
            # model = model.load_from_checkpoint(config.RESUME, strict=False)
            pretrained_state = torch.load(config.RESUME)["state_dict"]

            model_state = model.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size()}
            model.load_state_dict(pretrained_state, strict=False)

            # model_state = model.state_dict()
            # # # ignore_keys = ["epoch", "global_step",
            # # #                 "pytorch-lightning_version",
            # # #                 "state_dict", "callbacks",
            # # #                 "optimizer_states", "lr_schedulers",
            # # #                 "hparams_name", "hyper_parameters"]
            # # for k, v in pretrained_state.items():
            # #     if k in model_state: # v.size() == model_state[k].size():
            # #         pretrained_state[k] = v
            #
            # keep_keys = [ "state_dict"]
            # pretrained_state = {k: v for k, v in pretrained_state.items() if
            #                     k in model_state and v.size() == model_state[k].size() and k in keep_keys}
            # print(pretrained_state)
            # model_state.update(pretrained_state)
            # model.load_state_dict(model_state)


        #if not args.debug:
        #   wandb_logger.watch(model, log='all')

        print("In train loop....")
        # trainer.tune(model, LitDataModule)
        trainer.fit(model, LitDataModule)


if __name__ == "__main__":
    main()