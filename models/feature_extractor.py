"""
3D U-Net feature extraction network.

Input:  (B, 1, H, W, D) single-channel 3D volume
Output: (B, 128, H, W, D) 128-channel feature map preserving original spatial resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """3D convolution block: Conv3d + ReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            # nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ResBlock3D(nn.Module):
    """3D residual block: double convolution + skip connection, preserves resolution."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock3D(channels, channels),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            # nn.BatchNorm3d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = out + residual
        return self.relu(out)

class PixelShuffle3D(nn.Module):
    """3D pixel shuffle for upsampling spatial dimensions."""
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        r = self.upscale_factor
        b, c, d, h, w = x.size()

        if c % (r ** 3) != 0:
            raise ValueError(f"Input channels {c} must be divisible by r^3 = {r ** 3}")

        out_c = c // (r ** 3)

        x = x.view(b, out_c, r, r, r, d, h, w)

        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return x.view(b, out_c, d * r, h * r, w * r)        

class FeatureExtractor3D(nn.Module):
    """
    3D feature extraction network.

    A U-Net variant that preserves spatial resolution.

    Args:
        in_channels: Number of input channels (default=1).
        out_channels: Output feature dimension (default=128).
        base_channels: Base channel count (default=32).
        num_levels: Number of downsampling levels (default=3).
    """
    def __init__(self, in_channels=1, out_channels=128, base_channels=32, num_levels=3):
        super().__init__()
        
        self.num_levels = num_levels
        channels = [base_channels * (2**i) for i in range(num_levels + 1)]
        
        # ===== Encoder =====
        self.input_conv = ConvBlock3D(in_channels, base_channels)
        
        self.encoder_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        for i in range(num_levels):
            self.encoder_blocks.append(
                nn.Sequential(*[ResBlock3D(channels[i]) for _ in range(2)])
            )
            self.down_samples.append(
                nn.Conv3d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1)
            )
        

        self.bottleneck = nn.Sequential(
            *[ResBlock3D(channels[-1]) for _ in range(3)]
        )
        
   
        self.up_samples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(num_levels - 1, -1, -1):

            self.up_samples.append(
                nn.Sequential(
                    nn.ConvTranspose3d(channels[i+1], channels[i], kernel_size=2, stride=2),
                    # nn.BatchNorm3d(channels[i]),
                    nn.ReLU(inplace=True)
                )
            )
  
            self.decoder_blocks.append(
                nn.Sequential(
                    ConvBlock3D(channels[i] * 2, channels[i]),
                    ResBlock3D(channels[i])
                )
            )
        

        self.main_head = nn.Sequential(
            nn.Conv3d(base_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 1)
        )
        self.fine_head = nn.Sequential(
            nn.Conv3d(base_channels + 1, base_channels, 3, padding=1),
            nn.Conv3d(base_channels, base_channels * 8, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            PixelShuffle3D(upscale_factor=2),
            nn.Conv3d(base_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, out_channels, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    


    def forward(self, x):
        skips = []
        lr = x

        x = self.input_conv(x)
        

        for i in range(self.num_levels):
            x = self.encoder_blocks[i](x)
            skips.append(x)
            x = self.down_samples[i](x)
        
        x = self.bottleneck(x)
        
        for i in range(self.num_levels):
            x = self.up_samples[i](x)
            skip = skips[-(i+1)]
            
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            x = torch.cat([skip, x], dim=1)
            x = self.decoder_blocks[i](x)
        
        return self.main_head(x), self.fine_head(torch.cat([lr, x], dim=1))