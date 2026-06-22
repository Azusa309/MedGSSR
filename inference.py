import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import torch
import numpy as np

from models.medgssr import Net
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy import ndimage as nd
from utils import make_coord, extract_patches, reconstruct_volume
from tqdm import tqdm


def test(ckpt_path, input_path, save_path, s):
    """
    Run inference on a single 3D medical image.

    Args:
        ckpt_path: Path to model checkpoint.
        input_path: Path to input NIfTI file.
        save_path: Directory to save results.
        s: Scale factor.
    """

    device = torch.device('cuda')

    os.makedirs(save_path, exist_ok=True)
   
    PATCH_SIZE =  int(s * 32)
    STRIDE = PATCH_SIZE // 2
   
    # --- load model ---
    model = Net().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
   
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
   
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    with torch.no_grad():


        print(f"Processing sample: {input_path}")
       
        img = sitk.ReadImage(input_path)
        img_in = sitk.GetArrayFromImage(img).astype(np.float32)
        print(f"Original shape: {img_in.shape}")
        
        img_in = np.clip(img_in,a_min=-512, a_max=3071)

        min_val, max_val = np.min(img_in), np.max(img_in)
        if max_val - min_val > 0:
            img_in = (img_in - min_val) / (max_val - min_val)
       
        hr_shape = img_in.shape

        hr_patches_info = extract_patches(img_in, patch_size=PATCH_SIZE, stride=STRIDE)
        pred_patches = []  
       
        print(f"Starting inference on {len(hr_patches_info)} patches...")
        for i, (hr_patch_np, start_coord) in enumerate(tqdm(hr_patches_info)):

            lr_patch_np = nd.interpolation.zoom(hr_patch_np, 1 / s, order=3)
            lr_patch = torch.from_numpy(lr_patch_np).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)
           
            lr_points = make_coord((32, 32, 32), ranges=[[0,1],[0,1],[0,1]], flatten=False).unsqueeze(0).to(device)
            gs_points = make_coord((64, 64, 64), ranges=[[0,1],[0,1],[0,1]], flatten=False).unsqueeze(0).to(device)
            hr_points = make_coord((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), ranges=[[0,1],[0,1],[0,1]], flatten=False).unsqueeze(0).to(device)
            
            hr_pre_patch = model(lr_patch, lr_points, gs_points, hr_points)
            hr_pre_patch = torch.clamp(hr_pre_patch, 0.0, 1.0)
            pred_patches.append((hr_pre_patch[0,0].cpu().numpy(), start_coord))

        
        pred_full_np = reconstruct_volume(pred_patches, hr_shape, patch_size=PATCH_SIZE)
       
        pred_full = torch.from_numpy(pred_full_np).unsqueeze(0).unsqueeze(0).to(device)
        gt_full   = torch.from_numpy(img_in).unsqueeze(0).unsqueeze(0).to(device)
       
        mid_z = img_in.shape[0] // 2
        mid_y = img_in.shape[1] // 2
        mid_x = img_in.shape[2] // 2
       
        slice_configs = [
            (mid_z, 0, 'Axial'),     
            (mid_y, 1, 'Coronal'),    
            (mid_x, 2, 'Sagittal')    
        ]

        lpips_scores = []
       
        for slice_idx, axis, name in slice_configs:
            
            if axis == 0:
                pred_slice_2d = pred_full_np[slice_idx, :, :]
                gt_slice_2d   = img_in[slice_idx, :, :]
            elif axis == 1:
                pred_slice_2d = pred_full_np[:, slice_idx, :]
                gt_slice_2d   = img_in[:, slice_idx, :]
            else:
                pred_slice_2d = pred_full_np[:, :, slice_idx]
                gt_slice_2d   = img_in[:, :, slice_idx]
           
            plt.imsave(f'{save_path}/Pred_{name}.png', pred_slice_2d, cmap='gray')
            plt.imsave(f'{save_path}/GT_{name}.png', gt_slice_2d, cmap='gray')
           
       
        # Save NIfTI
        pred_sitk = sitk.GetImageFromArray(pred_full_np)
        pred_sitk.CopyInformation(img)
        sitk.WriteImage(pred_sitk, f"{save_path}/result.nii.gz")
       
        print(f"Done! Results have been saved to: {save_path}")



if __name__ == '__main__':
    ckpt_path = 'checkpoints_MSD_m4/model_epoch_550.pkl'
    input_path = "test_images/hcp.nii"
    save_path = "results"
    s = 3.5

    test(ckpt_path, input_path, save_path, s)