import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import argparse
from models.medgssr import Net
from dataset import MSD, HCP, MELA
import warnings
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import lpips
import torchmetrics

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

# ================== 3D FFT Loss ==================
class FFTLoss3D(nn.Module):
    """3D frequency-domain loss: L1 difference of magnitude spectra."""
    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred, target):
        # pred, target: (B, C, D, H, W)
        pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
        target_fft = torch.fft.fftn(target, dim=(-3, -2, -1))

        amp_pred = torch.abs(pred_fft)
        amp_target = torch.abs(target_fft)

        if self.loss_type == 'l1':
            return F.l1_loss(amp_pred, amp_target)
        else:
            return F.mse_loss(amp_pred, amp_target)

# ================== Metric Helper Functions ==================
def compute_psnr_3d(pred, target, data_range=1.0):
    """Compute PSNR (dB) for a 3D volume."""
    mse = F.mse_loss(pred, target)
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr.item()

def compute_ssim_3d(ssim_module, pred, target):
    """Compute 3D SSIM using torchmetrics (returns scalar)."""
    # pred/target shape (B, C, D, H, W), value range should match module.data_range
    ssim_module.update(pred, target)
    ssim_val = ssim_module.compute()
    ssim_module.reset()
    return ssim_val.item()

def compute_lpips_slice(lpips_fn, pred, target, slice_idx='mid'):
    """Compute LPIPS for a single slice (replicated to 3-channel)."""
    # pred, target: (B, C, D, H, W), C=1
    B, C, D, H, W = pred.shape
    if slice_idx == 'mid':
        idx = D // 2
    else:
        idx = slice_idx

    pred_slice = pred[:, :, idx, :, :]   # (B,1,H,W)
    target_slice = target[:, :, idx, :, :]

    # Convert to 3-channel (repeat color channels)
    pred_rgb = pred_slice.repeat(1, 3, 1, 1)
    target_rgb = target_slice.repeat(1, 3, 1, 1)

    # LPIPS expects input range [-1, 1] or [0, 1]; assuming [0, 1] here
    with torch.no_grad():
        lpips_val = lpips_fn(pred_rgb, target_rgb).mean().item()
    return lpips_val

# ================== Distributed Setup ==================
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# ================== Main Training Function ==================
def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(local_rank, world_size)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if local_rank == 0:
        print(f"Distributed training started on {world_size} GPUs")

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-hr_data_train', type=str, default='MELA', dest='hr_data_train')
    parser.add_argument('--resume', type=str, default='checkpoints_MELA_m4/model_epoch_500.pkl', help='path to latest checkpoint')
    parser.add_argument('-lr', type=float, default=1e-4, dest='lr')
    parser.add_argument('-lr_decay_epoch', type=int, default=100, dest='lr_decay_epoch')
    parser.add_argument('-epoch', type=int, default=800, dest='epoch')
    parser.add_argument('-summary_epoch', type=int, default=25, dest='summary_epoch')
    parser.add_argument('-bs', type=int, default=2, dest='batch_size')
    parser.add_argument('--fft_weight', type=float, default=0.1, help='weight for FFT loss')
    args = parser.parse_args()

    # ===== Dataset =====
    train_dataset = MELA(args.hr_data_train, batch_size=args.batch_size)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        persistent_workers=True
    )

    # ===== Fixed evaluation batch (rank 0 only) =====
    eval_data = None
    if local_rank == 0:
        # Take a fixed batch from the dataset without shuffling
        eval_dataset = MELA(args.hr_data_train, batch_size=args.batch_size)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
        eval_batch = next(iter(eval_loader))
        # Unpack and move to device
        eval_lr, eval_hr, eval_pts_lr, eval_pts_gs, eval_pts_hr = eval_batch
        eval_data = (
            eval_lr.unsqueeze(1).to(device, dtype=torch.float),
            eval_hr.unsqueeze(1).to(device, dtype=torch.float),
            eval_pts_lr.to(device, dtype=torch.float),
            eval_pts_gs.to(device, dtype=torch.float),
            eval_pts_hr.to(device, dtype=torch.float)
        )
        print("Fixed evaluation data loaded (rank0).")

    # ===== Model =====
    net = Net().to(device)
    net = DDP(net, device_ids=[local_rank])

    # ===== Optimizer =====
    params = [
        {"params": [p for n, p in net.named_parameters() if "encoder" in n], "lr": args.lr},
        {"params": [p for n, p in net.named_parameters() if "encoder" not in n], "lr": args.lr},
    ]
    optimizer = torch.optim.Adam(params)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    # ===== Resume from checkpoint =====
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume) and local_rank == 0:
        checkpoint = torch.load(args.resume, map_location=device)
        net.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"==> Resumed from epoch {start_epoch}")

    start_epoch_tensor = torch.tensor(start_epoch, device=device)
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = start_epoch_tensor.item()

    # ===== Loss Functions =====
    l1_loss_fn = nn.L1Loss()
    fft_loss_fn = FFTLoss3D(loss_type='l1').to(device)

    # ===== Metrics (rank0 only) =====
    if local_rank == 0:
        lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
        ssim_3d = torchmetrics.image.StructuralSimilarityIndexMeasure(
            kernel_size=3,
            data_range=1.0
        ).to(device)

    # ===== Log file =====
    log_file = None
    if local_rank == 0:
        log_file = open('training_log_m4_mela.txt', 'a', buffering=1)

    # ===== Training Loop =====
    for e in range(start_epoch, args.epoch):
        net.train()
        train_sampler.set_epoch(e)

        epoch_loss = 0.0

        for i, (patch_lr, patch_hr, lr_points, gs_points, hr_points) in enumerate(train_loader):
            patch_lr   = patch_lr.unsqueeze(1).to(device, dtype=torch.float)
            patch_hr   = patch_hr.unsqueeze(1).to(device, dtype=torch.float)
            hr_points  = hr_points.to(device, dtype=torch.float)
            lr_points  = lr_points.to(device, dtype=torch.float)
            gs_points  = gs_points.to(device, dtype=torch.float)

            hr_pre = net(patch_lr, lr_points, gs_points, hr_points)

            # Combined loss
            l1_loss = l1_loss_fn(hr_pre, patch_hr)
            fft_loss = fft_loss_fn(hr_pre, patch_hr)
            loss = l1_loss + args.fft_weight * fft_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            # Visualization (rank0 only, every 100 steps)
            if local_rank == 0 and (i + 1) % 100 == 0:
                os.makedirs('vis_train', exist_ok=True)
                with torch.no_grad():
                    img_pre = hr_pre[0, 0].cpu().numpy()
                    img_gt = patch_hr[0, 0].cpu().numpy()
                    mid_slice = img_pre.shape[0] // 2
                    slice_pre = img_pre[mid_slice, :, :]
                    slice_gt = img_gt[mid_slice, :, :]
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(slice_pre, cmap='gray')
                    axes[0].set_title(f'Pred (slice {mid_slice})')
                    axes[1].imshow(slice_gt, cmap='gray')
                    axes[1].set_title('GT')
                    plt.savefig(f'vis_train/E{e+1}_S{i+1}.png')
                    plt.close(fig)

        # Cross-GPU loss averaging
        epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(epoch_loss_tensor)
        avg_loss = epoch_loss_tensor.item() / len(train_loader) / world_size

        # ===== Log metrics every summary_epoch =====
        if local_rank == 0 and (e + 1) % args.summary_epoch == 0:
            net.eval()
            with torch.no_grad():
                # Use fixed evaluation samples
                if eval_data is not None:
                    eval_lr, eval_hr, eval_pts_lr, eval_pts_gs, eval_pts_hr = eval_data
                    eval_pre = net.module(eval_lr, eval_pts_lr, eval_pts_gs, eval_pts_hr)

                # 3D PSNR
                psnr_val = compute_psnr_3d(eval_pre, eval_hr, data_range=1.0)

                # 3D SSIM
                ssim_val = compute_ssim_3d(ssim_3d, eval_pre, eval_hr)

                # Slice LPIPS (mid-slice)
                lpips_val = compute_lpips_slice(lpips_fn, eval_pre, eval_hr, slice_idx='mid')

                    # Print and log
                    metric_line = (f"Epoch {e+1}/{args.epoch} | "
                                   f"Loss: {avg_loss:.6f} | "
                                   f"PSNR: {psnr_val:.3f} dB | "
                                   f"SSIM: {ssim_val:.4f} | "
                                   f"LPIPS(slice): {lpips_val:.4f}\n")
                    print(metric_line.strip())
                    log_file.write(metric_line)
                    log_file.flush()

        # Save checkpoint
        if local_rank == 0 and (e + 1) % args.summary_epoch == 0:
            os.makedirs('checkpoints_MELA_m4', exist_ok=True)
            torch.save({
                'epoch': e + 1,
                'model_state_dict': net.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'checkpoints_MELA_m4/model_epoch_{e + 1}.pkl')

        # Learning rate decay
        if (e + 1) % args.lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            if local_rank == 0:
                print(f"Learning rate decayed to: {optimizer.param_groups[0]['lr']}")

    if local_rank == 0:
        log_file.close()
        print("Training finished.")
    cleanup()

if __name__ == '__main__':
    main()