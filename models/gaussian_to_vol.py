"""
Gaussian-to-Volume rendering module.

Renders a set of 3D Gaussians into a volumetric intensity grid
using the CUDA-accelerated discretized splatting kernel.
"""
import torch
import torch.nn as nn
from gs_utils.Compute_intensity import compute_intensity as compute_intensity_cuda


class GaussianToVol(nn.Module):
    """
    Render Gaussian primitives to a 3D volume.

    Computes per-voxel intensity by splatting Gaussians onto a
    target coordinate grid using inverse covariance matrices.
    """
        super().__init__()
    
    def get_inv_covariance(self, scaling, rotations):
        scaling_inv_squared = 1.0 / (scaling ** 2 + 1e-7)
        S_inv_squared = torch.diag_embed(scaling_inv_squared)
        R_transpose = rotations.transpose(-1, -2)
        covariance_inv = torch.matmul(rotations, torch.matmul(S_inv_squared, R_transpose))
        return covariance_inv

    def single_batch_render(self, hr_points_single, means, intensity, scales, rotations):
        D_hr, H_hr, W_hr = hr_points_single.shape[1:4]
 
        gaussian_centers = means.contiguous()
        intensities = intensity.contiguous() 
        scales_flat = scales.contiguous()
        
        inv_covariance = self.get_inv_covariance(scales, rotations) 
        inv_covariances = inv_covariance.view(-1, 9).contiguous()

        grid_points = hr_points_single[0].view(-1, 3).contiguous()

        intensity_vol = torch.zeros(1, D_hr, H_hr, W_hr, device=hr_points_single.device).contiguous()

        intensity_vol = compute_intensity_cuda(
            gaussian_centers,
            grid_points,
            intensities,
            inv_covariances,
            scales_flat,
            intensity_vol
        )

        return intensity_vol.unsqueeze(-1)

    def forward(self, hr_points, gaussians):
        means_list     = gaussians['means']         
        intensity_list = gaussians['intensity']
        scales_list    = gaussians['scales']
        rotations_list = gaussians['rotations']

        B, D_hr, H_hr, W_hr = hr_points.shape[0:4]
        intensity_grid = torch.zeros(B, D_hr, H_hr, W_hr, 1, device=hr_points.device)

        for i in range(B):
            means      =   means_list[i]
            intensity  =   intensity_list[i]
            scales     =   scales_list[i]
            rotations  =   rotations_list[i]
            
            single_vol = self.single_batch_render(
                hr_points[i:i+1],      
                means,
                intensity,
                scales,
                rotations
            )

            intensity_grid[i] = single_vol[0]  

        return intensity_grid.permute(0, 4, 1, 2, 3)