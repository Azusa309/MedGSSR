import os
import random
import torch
import numpy as np
from torch.utils import data
from scipy import ndimage as nd
import utils
import nibabel as nib


class MSD(data.Dataset):
    """
    MSD (Medical Segmentation Decathlon) dataset for training.

    Loads .npy patches and applies random cropping with a random scale factor.
    """
    def __init__(self, path, s_range=[2, 4], batch_size=12):
        self.path = path
        self.s_range = s_range
        self.batch_size = batch_size
        self.filenames = sorted([f for f in os.listdir(self.path) if f.endswith('.npy')])

        self.call_count = 0
        self.last_s = 2.0

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # Update scale factor only at batch boundaries so all samples in a batch share the same scale
        if self.call_count % self.batch_size == 0:
            self.last_s = np.round(random.uniform(self.s_range[0], self.s_range[1] + 0.04), 1)

        s = self.last_s
        self.call_count += 1

        # Load and normalize
        patch_full = np.load(os.path.join(self.path, self.filenames[index])).astype(np.float32)
        p_min, p_max = np.min(patch_full), np.max(patch_full)
        if p_max > p_min:
            patch_full = (patch_full - p_min) / (p_max - p_min)

        # Crop and scale
        base_size = np.array([32, 32, 32])

        d_size, h_size, w_size = (base_size * s).astype(int)
        d_full, h_full, w_full = patch_full.shape

        max_d_start = d_full - d_size
        max_h_start = h_full - h_size
        max_w_start = w_full - w_size

        d_start = np.random.randint(0, max_d_start + 1)
        h_start = np.random.randint(0, max_h_start + 1)
        w_start = np.random.randint(0, max_w_start + 1)

        hr_patch = patch_full[
            d_start:d_start + d_size,
            h_start:h_start + h_size,
            w_start:w_start + w_size
        ]
        lr_patch = nd.interpolation.zoom(hr_patch, 1 / s, order=3)

        # Generate coordinates
        lr_points = utils.make_coord(lr_patch.shape, ranges=[[0, 1], [0, 1], [0, 1]], flatten=False)
        hr_points = utils.make_coord(hr_patch.shape, ranges=[[0, 1], [0, 1], [0, 1]], flatten=False)
        gs_points = utils.make_coord((64, 64, 64), ranges=[[0, 1], [0, 1], [0, 1]], flatten=False)
        return (torch.from_numpy(lr_patch),
                torch.from_numpy(hr_patch),
                lr_points,
                gs_points,
                hr_points)


class MSD_ForTest(data.Dataset):
    """
    MSD dataset for testing/inference.

    Uses a fixed scale factor and deterministic cropping.
    """
    def __init__(self, path, s=4):
        self.path = path
        self.s = s
        self.filenames = [f for f in os.listdir(self.path) if f.endswith('.npy')]

    def __getitem__(self, index):
        patch_full = np.load(os.path.join(self.path, self.filenames[index])).astype(np.float32)

        p_min = np.min(patch_full)
        p_max = np.max(patch_full)

        patch_full = (patch_full - p_min) / (p_max - p_min)

        base_size = np.array([32, 32, 32])
        d_size, h_size, w_size = (base_size * self.s).astype(int)

        hr_patch = patch_full[:d_size, :h_size, :w_size]
        lr_patch = nd.interpolation.zoom(hr_patch, 1 / self.s, order=3, grid_mode=True)

        hr_points = utils.make_coord(hr_patch.shape, ranges=[[0, 1], [0, 1], [0, 1]], flatten=False)
        lr_points = utils.make_coord((32, 32, 32), ranges=[[0, 1], [0, 1], [0, 1]], flatten=False)
        gs_points = utils.make_coord((64, 64, 64), ranges=[[0, 1], [0, 1], [0, 1]], flatten=False)

        return lr_patch, hr_patch, lr_points, gs_points, hr_points

    def __len__(self):
        return len(self.filenames)


class MELA(data.Dataset):
    """
    MELA dataset for training.

    Loads .nii.gz volumes and applies random cropping with a random scale factor.
    """
    def __init__(self, path='MELA', s_range=[2, 4], batch_size=12):
        self.path = path
        self.s_range = s_range
        self.batch_size = batch_size
        self.filenames = sorted([f for f in os.listdir(self.path) if f.endswith('.nii.gz')])

        self.call_count = 0
        self.last_s = 2.0

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        if self.call_count % self.batch_size == 0:
            self.last_s = np.round(random.uniform(self.s_range[0], self.s_range[1] + 0.04), 1)

        s = self.last_s
        self.call_count += 1

        img = nib.load(os.path.join(self.path, self.filenames[index]))
        data = img.get_fdata().astype(np.float32)
        patch_full = np.clip(data, a_min=-512, a_max=3071)
        p_min, p_max = np.min(patch_full), np.max(patch_full)
        if p_max > p_min:
            patch_full = (patch_full - p_min) / (p_max - p_min)

        base_size = np.array([32, 32, 32])

        d_size, h_size, w_size = (base_size * s).astype(int)
        d_full, h_full, w_full = patch_full.shape

        max_d_start = d_full - d_size
        max_h_start = h_full - h_size
        max_w_start = w_full - w_size

        d_start = np.random.randint(0, max_d_start + 1)
        h_start = np.random.randint(0, max_h_start + 1)
        w_start = np.random.randint(0, max_w_start + 1)

        hr_patch = patch_full[
            d_start:d_start + d_size,
            h_start:h_start + h_size,
            w_start:w_start + w_size
        ]

        lr_patch = nd.interpolation.zoom(hr_patch, 1 / s, order=3)

        lr_points = utils.make_coord(lr_patch.shape, ranges=[[0, 1], [0, 1], [0, 1]], flatten=False)
        hr_points = utils.make_coord(hr_patch.shape, ranges=[[0, 1], [0, 1], [0, 1]], flatten=False)
        gs_points = utils.make_coord((64, 64, 64), ranges=[[0, 1], [0, 1], [0, 1]], flatten=False)
        return (torch.from_numpy(lr_patch),
                torch.from_numpy(hr_patch),
                lr_points,
                gs_points,
                hr_points)