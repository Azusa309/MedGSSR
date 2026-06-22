import torch.nn as nn
import torch
from models.feature_extractor import FeatureExtractor3D
from models.gaussian_to_vol import GaussianToVol
from models.feature_to_gaussian import FeatureToGaussian
from utils import merge_gaussian_dicts


class Net(nn.Module):
    """
    MedGSSR: Medical Gaussian Splatting for Super-Resolution.

    Combines a 3D U-Net encoder with Gaussian splatting-based rendering
    for anisotropic medical image super-resolution.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = FeatureExtractor3D()
        self.FeatureToGaussian = FeatureToGaussian(in_channels=128, num_sub_voxels=4)
        self.GaussianToVol = GaussianToVol()

    def forward(self, lr_patch, lr_points, gs_points, hr_points):
        features_c, features_f = self.encoder(lr_patch)                       
        gaussians_c = self.FeatureToGaussian(features_c, lr_points, 4, 2)              
        gaussians_f = self.FeatureToGaussian(features_f, gs_points, 6, 4)       
        gaussians = merge_gaussian_dicts(gaussians_c, gaussians_f)
        hr_pre = self.GaussianToVol(hr_points, gaussians)

        return hr_pre