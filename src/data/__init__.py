"""
Author: Vaishakh Patil
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

from .NYU_dataset import Nyu2DataModule, Nyu2DataModuleTest
from .KITTI_dataset import KITTIDataModule, KITTIDataModuleTest

dataset_dict = {'NYU_Depth_v2': Nyu2DataModule, 'KITTI_eigen': KITTIDataModule, 'KITTI_eigen_Test': KITTIDataModuleTest, 'NYU_Depth_v2_Test': Nyu2DataModuleTest}

def allowed_dataset():
    return dataset_dict.keys()

def define_dataset(dataset, args):
    if dataset not in allowed_dataset():
        raise KeyError("The requested dataset: {} is not implemented".format(dataset))
    else:
        return dataset_dict[dataset](args)