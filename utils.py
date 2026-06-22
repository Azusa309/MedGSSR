"""
Utility functions for MedGSSR.

Provides coordinate grid generation and cached image I/O.
"""
import os
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def read_img(in_path):
    """Read all images from a directory, with caching support."""
    if 'train' in in_path.lower():
        cache_name = "train_data_cache.pt"
    elif 'val' in in_path.lower():
        cache_name = "val_data_cache.pt"
    else:
        cache_name = "data_cache.pt"

    if os.path.exists(cache_name):
        print(f"Loading cached data from: {cache_name}")
        return torch.load(cache_name)

    img_list = []
    filenames = sorted(os.listdir(in_path))
    print(f"Reading raw data for {cache_name} ...")
    for f in tqdm(filenames):
        img = sitk.ReadImage(os.path.join(in_path, f))
        img_vol = sitk.GetArrayFromImage(img)
        img_list.append(img_vol)

    torch.save(img_list, cache_name)
    print(f"Cache saved to: {cache_name}")

    return img_list

# Adapted from LIIF (https://github.com/yinboc/liif/blob/main/utils.py)
def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers.

    Args:
        shape: tuple of ints, e.g. (40, 40, 40).
        ranges: list of [v0, v1] per dimension. Defaults to [0, 1].
        flatten: if True, flatten to (N, 3); else return (D, H, W, 3).

    Returns:
        Coordinate tensor.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = 0, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def extract_patches(volume, patch_size, stride):
    """
    Extract overlapping patches from a 3D volume.

    Args:
        volume: numpy array [D, H, W].
        patch_size: size of each cubic patch.
        stride: stride between patches.

    Returns:
        list of (patch, (z_start, y_start, x_start)).
    """
    D, H, W = volume.shape
    patches = []

    z_starts = list(range(0, max(D - patch_size + 1, 1), stride))
    y_starts = list(range(0, max(H - patch_size + 1, 1), stride))
    x_starts = list(range(0, max(W - patch_size + 1, 1), stride))

    if z_starts and z_starts[-1] + patch_size < D:
        z_starts.append(D - patch_size)
    if y_starts and y_starts[-1] + patch_size < H:
        y_starts.append(H - patch_size)
    if x_starts and x_starts[-1] + patch_size < W:
        x_starts.append(W - patch_size)

    for z_start in z_starts:
        for y_start in y_starts:
            for x_start in x_starts:
                z_end = min(z_start + patch_size, D)
                y_end = min(y_start + patch_size, H)
                x_end = min(x_start + patch_size, W)

                patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]

                if patch.shape != (patch_size, patch_size, patch_size):
                    pad_z = patch_size - patch.shape[0]
                    pad_y = patch_size - patch.shape[1]
                    pad_x = patch_size - patch.shape[2]
                    patch = np.pad(patch, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')

                patches.append((patch, (z_start, y_start, x_start)))

    return patches


def reconstruct_volume(pred_patches, target_shape, patch_size=64):
    """
    Reconstruct a full volume from overlapping predicted patches.

    Args:
        pred_patches: list of (patch_array, (z_start, y_start, x_start)).
        target_shape: (D, H, W) of the output volume.
        patch_size: size of each cubic patch.

    Returns:
        Reconstructed numpy array of shape target_shape.
    """
    D, H, W = target_shape
    recon = np.zeros((D, H, W), dtype=np.float32)
    weight_sum = np.zeros((D, H, W), dtype=np.float32)

    for pred_patch, (z_s, y_s, x_s) in pred_patches:
        dz, dy, dx = [min(s + patch_size, org) - s for s, org in zip((z_s, y_s, x_s), (D, H, W))]
        recon[z_s:z_s + dz, y_s:y_s + dy, x_s:x_s + dx] += pred_patch[:dz, :dy, :dx]
        weight_sum[z_s:z_s + dz, y_s:y_s + dy, x_s:x_s + dx] += 1.0

    mask = weight_sum > 1e-6
    recon[mask] /= weight_sum[mask]
    return recon


def merge_gaussian_dicts(dict1, dict2):
    """
    Merge two Gaussian parameter dictionaries by concatenation.

    Args:
        dict1: First dict with keys "means", "scales", "rotations", "intensity".
        dict2: Second dict with the same keys.

    Returns:
        Merged dictionary.
    """
    return {
        "means": torch.cat([dict1["means"], dict2["means"]], dim=1),
        "scales": torch.cat([dict1["scales"], dict2["scales"]], dim=1),
        "rotations": torch.cat([dict1["rotations"], dict2["rotations"]], dim=1),
        "intensity": torch.cat([dict1["intensity"], dict2["intensity"]], dim=1),
    }