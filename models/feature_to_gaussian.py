"""
Feature-to-Gaussian module.

Converts 3D feature maps into Gaussian primitive parameters
(means, scales, rotations, intensity) via MLPs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureToGaussian(nn.Module):
    """
    Map feature vectors to Gaussian parameters.

    Each voxel produces n sub-voxel Gaussians with learned offsets,
    scales, rotations, and intensities.

    Args:
        in_channels: Feature dimension (default=128).
        num_sub_voxels: Number of Gaussians per voxel (default=4).
    """
    def __init__(self, in_channels=128, num_sub_voxels=4):
        super().__init__()
        self.in_dim = in_channels
        self.num_sub_voxels = num_sub_voxels
        self.offset_mlp = self.make_mlp(128, 3 * self.num_sub_voxels, hidden_dim=[256, 512, 256])
        self.scale_mlp = self.make_mlp(128, 3 * self.num_sub_voxels, hidden_dim=[256, 512, 256])
        self.rotation_mlp = self.make_mlp(128, 4 * self.num_sub_voxels, hidden_dim=[256, 512, 256])
        self.opacity_mlp = self.make_mlp(128, 1 * self.num_sub_voxels, hidden_dim=[256, 512, 256])

    def quaternion_to_rotation_matrix(self, q):
        """Convert unit quaternion (..., 4) to rotation matrix (..., 3, 3)."""
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        batch_shape = q.shape[:-1]

        R = torch.empty((*batch_shape, 3, 3), device=q.device)

        R[..., 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
        R[..., 0, 1] = 2 * (x * y - z * w)
        R[..., 0, 2] = 2 * (x * z + y * w)

        R[..., 1, 0] = 2 * (x * y + z * w)
        R[..., 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
        R[..., 1, 2] = 2 * (y * z - x * w)

        R[..., 2, 0] = 2 * (x * z - y * w)
        R[..., 2, 1] = 2 * (y * z + x * w)
        R[..., 2, 2] = 1 - 2 * (x ** 2 + y ** 2)

        return R

    @staticmethod
    def make_mlp(in_dim, out_dim, hidden_dim):
        """Build an MLP with GELU activations and Xavier init."""
        layers = []
        for h_in, h_out in zip([in_dim] + hidden_dim[:-1], hidden_dim):
            layers.extend([nn.Linear(h_in, h_out), nn.GELU()])
        layers.append(nn.Linear(hidden_dim[-1], out_dim))
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        return nn.Sequential(*layers)

    def forward(self, feat, lr_points, s, i):
        """
        Forward pass.

        Args:
            feat: Feature map (B, C, D, H, W).
            lr_points: Coordinate grid (B, D, H, W, 3).
            s: Scale bias for softplus.
            i: Intensity bias for sigmoid.

        Returns:
            dict with keys "means", "scales", "rotations", "intensity".
        """
        B, C_lr, D_lr, H_lr, W_lr = feat.shape
        feat_flat = feat.permute(0, 2, 3, 4, 1).reshape(-1, C_lr)

        lr_points_expanded = lr_points.unsqueeze(4)
        lr_points_expanded = lr_points_expanded.expand(-1, -1, -1, -1, self.num_sub_voxels, -1)

        off_raw = self.offset_mlp(feat_flat).view(B, D_lr, H_lr, W_lr, self.num_sub_voxels, 3)
        scale_raw = self.scale_mlp(feat_flat).view(B, -1, 3)
        rot_raw = self.rotation_mlp(feat_flat).view(B, -1, 4)
        intensity_raw = self.opacity_mlp(feat_flat).view(B, -1, 1)

        means = torch.clamp(lr_points_expanded + torch.tanh(off_raw) / D_lr, min=0, max=1).view(B, -1, 3)
        scales = torch.clamp(F.softplus(scale_raw - s), min=1e-10, max=3)
        R = self.quaternion_to_rotation_matrix(F.normalize(rot_raw, p=2, dim=-1))
        intensity = torch.sigmoid(intensity_raw - i)

        return {
            "means": means,
            "scales": scales,
            "rotations": R,
            "intensity": intensity,
        }