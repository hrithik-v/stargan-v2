"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.wing import FAN

# --- StyleGAN2 Inspired Helper Modules ---
class Blur(nn.Module):
    def __init__(self, kernel_1d, pad, upsample_factor=1, channels=0): # channels needed for grouped conv
        super().__init__()
        kernel = torch.tensor(kernel_1d, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = kernel[:, None] * kernel[None, :] # Convert to 2D
        kernel /= kernel.sum()
        if upsample_factor > 1: # Not typically used like this in StyleGAN2 blur, factor is for main op
            kernel = kernel * (upsample_factor ** 2)
        
        # Store kernel for dynamic channel repetition
        self.kernel_base = kernel 
        self.pad = pad
        # self.channels = channels # Will be set in forward

    def forward(self, x):
        channels = x.size(1)
        kernel = self.kernel_base.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1).to(x.device)
        return F.conv2d(x, kernel, padding=self.pad, groups=channels)

class UpSample(nn.Module):
    def __init__(self, kernel_1d=[1, 3, 3, 1], factor=2):
        super().__init__()
        self.factor = factor
        self.pad = (len(kernel_1d) - 1) // 2 # Assuming symmetric kernel for blur
        self.blur = Blur(kernel_1d, self.pad)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        x = self.blur(x)
        return x

class DownSample(nn.Module):
    def __init__(self, kernel_1d=[1, 3, 3, 1], factor=2):
        super().__init__()
        self.factor = factor
        self.pad = (len(kernel_1d) - 1) // 2 # Assuming symmetric kernel for blur
        self.blur = Blur(kernel_1d, self.pad)

    def forward(self, x):
        x = self.blur(x)
        x = F.avg_pool2d(x, kernel_size=self.factor, stride=self.factor)
        return x

# --- Non-Styled Residual Block (for Sseg) ---
class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, normalize=False, downsample=False, upsample=False, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.upsample = upsample
        self.learned_sc = (dim_in != dim_out) or downsample or upsample

        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_out, affine=True)
        else: # If not normalizing, use identity layers
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        if self.learned_sc:
            self.conv_sc = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        else:
            self.conv_sc = nn.Identity()

    def _shortcut(self, x):
        res_x = x
        if self.upsample:
            res_x = F.interpolate(res_x, scale_factor=2, mode='nearest')
        res_x = self.conv_sc(res_x)
        if self.downsample: # Note: upsample and downsample are mutually exclusive
            res_x = F.avg_pool2d(res_x, 2)
        return res_x

    def _residual(self, x):
        res_x = x
        res_x = self.norm1(res_x)
        res_x = self.actv(res_x)
        if self.upsample:
             res_x = F.interpolate(res_x, scale_factor=2, mode='nearest')
        res_x = self.conv1(res_x)
        
        if self.downsample:
            res_x = F.avg_pool2d(res_x, 2)
        
        res_x = self.norm2(res_x)
        res_x = self.actv(res_x)
        res_x = self.conv2(res_x)
        return res_x

    def forward(self, x):
        return self._shortcut(x) + self._residual(x)

# --- High Pass Filter ---
class HighPass(nn.Module):
    def __init__(self, w_hpf, device='cpu'): # Device can be updated in forward
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]], dtype=torch.float32) / w_hpf
        # Repeat for 3 channels, ensure it's (out_channels, in_channels/groups, kH, kW)
        # For depthwise separable, groups = in_channels. Here, 3 groups for 3 input channels.
        self.filter = self.filter.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1) 
        self.w_hpf = w_hpf

    def forward(self, x):
        if self.filter.device != x.device:
            self.filter = self.filter.to(x.device)
        # Apply convolution with groups=3 for 3-channel images if filter is (3,1,k,k)
        # If filter is (1,1,k,k) and repeated for conv2d, groups=1 is fine and conv2d applies it to each input channel if out_channels=3.
        # Given self.filter.repeat(3,1,1,1), it becomes (3,1,3,3). This is suitable for groups=3 if input has 3 channels.
        return F.conv2d(x, self.filter, padding=1, groups=3) # groups=3 for per-channel filtering


# --- StyleGAN2-inspired components for weight demodulation ---
class StyledConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True, upsample=False, downsample=False, activation=nn.LeakyReLU(0.2), blur_kernel_1d=[1, 3, 3, 1]):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=True) # Bias can be true or handled separately
        self.affine = nn.Linear(style_dim, in_channels)
        self.demodulate = demodulate
        self.upsample_op = upsample
        self.downsample_op = downsample
        self.activation = activation

        if upsample:
            self.upsampler = UpSample(blur_kernel_1d)
        if downsample:
            self.downsampler = DownSample(blur_kernel_1d)
        
        # Optional: separate bias after demodulation, StyleGAN2 often does this
        # self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x, style):
        batch_size, in_channel_x, height, width = x.shape # in_channel_x should match self.conv.in_channels
        
        # Affine transformation of style code
        style = self.affine(style).view(batch_size, 1, self.conv.in_channels, 1, 1) # B x 1 x C_in x 1 x 1

        # Modulate weights
        # Weight shape: (C_out, C_in, K, K)
        weight = self.conv.weight.unsqueeze(0) # 1 x C_out x C_in x K x K
        weight = weight * style # B x C_out x C_in x K x K (broadcasting style)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8) # B x C_out
            weight = weight * demod.view(batch_size, self.conv.out_channels, 1, 1, 1) # B x C_out x C_in x K x K
        
        # Reshape for grouped convolution
        # Input x: (B, C_in, H, W)
        # We need to make it (1, B * C_in, H, W) for grouped conv
        # Weight: (B * C_out, C_in, K, K)
        
        if self.upsample_op:
            x = self.upsampler(x)
            # Update height/width after upsampling for the reshape
            _, _, height, width = x.shape 
        
        if self.downsample_op:
            x = self.downsampler(x)
            # Update height/width after downsampling
            _, _, height, width = x.shape

        # Prepare for grouped convolution
        x = x.reshape(1, batch_size * self.conv.in_channels, height, width)
        weight = weight.view(
            batch_size * self.conv.out_channels, self.conv.in_channels, self.conv.kernel_size[0], self.conv.kernel_size[1]
        )

        out = F.conv2d(x, weight, padding=self.conv.padding[0], groups=batch_size)
        out = out.view(batch_size, self.conv.out_channels, *out.shape[2:])

        if self.conv.bias is not None: # Add original conv bias if it exists
            out = out + self.conv.bias.view(1, -1, 1, 1)
        # Alternatively, add separate learned bias: out = out + self.bias
        
        if self.activation:
            out = self.activation(out)
            
        return out

class WeightDemodResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim, upsample=False, downsample=False, activation=nn.LeakyReLU(0.2), blur_kernel_1d=[1,3,3,1]):
        super().__init__()
        self.styled_conv1 = StyledConv2d(dim_in, dim_out, 3, style_dim, upsample=upsample, downsample=downsample, activation=activation, blur_kernel_1d=blur_kernel_1d)
        self.styled_conv2 = StyledConv2d(dim_out, dim_out, 3, style_dim, activation=activation, blur_kernel_1d=blur_kernel_1d)
        
        self.skip_connection = None
        if dim_in != dim_out or upsample or downsample:
            # Skip connection also needs to be styled if it involves convolution, or be a simple up/downsample + 1x1
            # For simplicity, using a StyledConv2d with kernel 1 for skip, no demodulation on skip usually
            self.skip_connection = StyledConv2d(dim_in, dim_out, 1, style_dim, demodulate=False, upsample=upsample, downsample=downsample, activation=None, blur_kernel_1d=blur_kernel_1d)

    def forward(self, x, s): # s is the style vector
        skip_val = x
        if self.skip_connection:
            skip_val = self.skip_connection(skip_val, s)
        
        out = self.styled_conv1(x, s)
        out = self.styled_conv2(out, s)
        
        return (out + skip_val) / math.sqrt(2) # Normalize as in StyleGAN2


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=0, num_domains=5, num_seg_classes=19): # Added num_seg_classes
        super().__init__()
        self.img_size = img_size
        self.style_dim = style_dim
        self.num_domains = num_domains
        self.w_hpf = w_hpf
        self.num_seg_classes = num_seg_classes # For Sseg output

        initial_conv_dim = 2**14 // img_size
        self.from_rgb = nn.Sequential(
            nn.Conv2d(3, initial_conv_dim, 3, 1, 1),
            nn.InstanceNorm2d(initial_conv_dim, affine=True),
            nn.LeakyReLU(0.2),
        )

        self.shared_encoder_down = nn.ModuleList()
        self.shared_encoder_bottleneck = nn.ModuleList()
        
        num_downsamples_encoder = int(np.log2(img_size / 16))
        enc_dim_in = initial_conv_dim
        self.encoder_skip_dims = [] 

        for _ in range(num_downsamples_encoder):
            dim_out = min(enc_dim_in * 2, max_conv_dim)
            self.shared_encoder_down.append(
                ResBlk(enc_dim_in, dim_out, normalize=True, downsample=True))
            enc_dim_in = dim_out
            self.encoder_skip_dims.append(enc_dim_in) 
        
        for _ in range(2): 
            self.shared_encoder_bottleneck.append(
                ResBlk(enc_dim_in, enc_dim_in, normalize=True))
        
        shared_bottleneck_dim = enc_dim_in

        # Sseg Decoder
        self.sseg_decoder = SsegDecoder(
            bottleneck_dim=shared_bottleneck_dim,
            num_upsamples=num_downsamples_encoder,
            encoder_skip_dims=self.encoder_skip_dims[::-1], # Reversed for decoder
            initial_conv_dim=initial_conv_dim,
            num_seg_classes=self.num_seg_classes,
            num_domains=self.num_domains
        )

        # Wclues Decoder
        self.wclues_decoder = WcluesDecoder(
            sseg_map_channels=self.num_seg_classes, # Input from SsegDecoder's map
            style_dim=self.style_dim 
            # Potentially add other params like hidden_dim, num_conv_blocks if made configurable
        )

        # Gglo Decoder (formerly WcluesDecoder)
        self.gglo_decoder = GgloDecoder(
            bottleneck_dim_in=shared_bottleneck_dim, 
            style_dim=style_dim, 
            num_upsamples=num_downsamples_encoder,
            encoder_skip_dims_reversed=self.encoder_skip_dims[::-1], 
            target_final_dim=initial_conv_dim
        )

        if self.w_hpf > 0:
            self.hpf = HighPass(self.w_hpf)

    def forward(self, x_original, s_style, masks=None, Sseg_only=False, **kwargs):
        processed_x = x_original
        if self.w_hpf > 0 and hasattr(self, 'hpf'):
            if self.hpf.filter.device != x_original.device:
                 self.hpf.filter = self.hpf.filter.to(x_original.device)
            processed_x = self.hpf(x_original)

        initial_features = self.from_rgb(processed_x)
        
        encoder_skips_for_decoders = []
        current_f = initial_features
        for block in self.shared_encoder_down:
            current_f = block(current_f)
            encoder_skips_for_decoders.append(current_f)
        
        bottleneck_f = current_f
        for block in self.shared_encoder_bottleneck:
            bottleneck_f = block(bottleneck_f)
        
        # Sseg Path
        # SsegDecoder might use skips if designed that way, pass encoder_skips_for_decoders[::-1]
        segmentation_map, weather_classification_logits = self.sseg_decoder(bottleneck_f, encoder_skips_for_decoders[::-1])

        if Sseg_only:
            return segmentation_map, weather_classification_logits

        # Wclues Path
        # WcluesDecoder takes Sseg's segmentation_map and the style s_style
        Wclues_blending_map = self.wclues_decoder(segmentation_map.detach(), s_style) # Detach if Sseg is not to be trained through Wclues directly

        # Gglo Path
        # GgloDecoder takes bottleneck_f, s_style, and reversed encoder skips
        Gglo_image = self.gglo_decoder(bottleneck_f, s_style, encoder_skips_for_decoders[::-1])
        
        # Final Image Synthesis (Equation 1)
        # G(x,w) = Wclues(x,w) * Gglo(x,w) + (1 - Wclues(x,w)) * x_original
        final_output_image = Wclues_blending_map * Gglo_image + (1 - Wclues_blending_map) * x_original
        
        # Return all necessary outputs for loss calculation during training
        # For inference, only final_output_image might be needed.
        return final_output_image, segmentation_map, weather_classification_logits, bottleneck_f


class SsegDecoder(nn.Module):
    def __init__(self, bottleneck_dim, num_upsamples, encoder_skip_dims, initial_conv_dim, num_seg_classes, num_domains):
        super().__init__()
        self.num_seg_classes = num_seg_classes
        self.decode_blocks = nn.ModuleList()
        self.skip_align_convs = nn.ModuleList() # For U-Net style skips if used

        curr_dim = bottleneck_dim
        # Note: Sseg is domain-invariant, so ResBlks here are not styled.
        for i in range(num_upsamples):
            dim_out_block = max(curr_dim // 2, initial_conv_dim) if i < num_upsamples - 1 else initial_conv_dim
            
            # Using ResBlk for Sseg decoder as it's domain-invariant
            self.decode_blocks.append(
                ResBlk(curr_dim, dim_out_block, normalize=True, upsample=True)
            )
            
            # If using U-Net skips for Sseg decoder (optional, paper doesn't detail Sseg decoder structure heavily)
            # current_encoder_skip_dim = encoder_skip_dims[i] # encoder_skip_dims should be reversed for decoder
            # self.skip_align_convs.append(nn.Conv2d(current_encoder_skip_dim, dim_out_block, 1))
            
            curr_dim = dim_out_block

        self.to_segmentation_map = nn.Conv2d(curr_dim, num_seg_classes, 3, 1, 1) # Output num_seg_classes channels

        # Classification head for Lc (weather type classification from Sseg features)
        # Takes features from the Sseg decoder's output (before to_segmentation_map)
        self.sseg_classifier_pool = nn.AdaptiveAvgPool2d(1)
        self.sseg_classifier_fc = nn.Linear(curr_dim, num_domains)

    def forward(self, bottleneck_features, encoder_skip_features_reversed=None):
        x = bottleneck_features
        for i, block in enumerate(self.decode_blocks):
            x = block(x)
            # Example if Sseg decoder used U-Net skips:
            # if encoder_skip_features_reversed and i < len(encoder_skip_features_reversed):
            #     skip_f = encoder_skip_features_reversed[i]
            #     if x.shape[2:] != skip_f.shape[2:]:
            #         skip_f = F.interpolate(skip_f, size=x.shape[2:], mode='bilinear', align_corners=False)
            #     aligned_skip = self.skip_align_convs[i](skip_f)
            #     x = x + aligned_skip
        
        segmentation_map = self.to_segmentation_map(x)
        
        # For Lc: weather classification
        pooled_features = self.sseg_classifier_pool(x).view(x.size(0), -1)
        weather_classification_logits = self.sseg_classifier_fc(pooled_features)
        
        return segmentation_map, weather_classification_logits

class WcluesDecoder(nn.Module):
    def __init__(self, sseg_map_channels, style_dim, hidden_dim=256, num_conv_blocks=2):
        super().__init__()
        # This decoder takes Sseg(x) and w, outputs a 1-channel blending map.
        # Architecture can be relatively simple.
        layers = [nn.Conv2d(sseg_map_channels, hidden_dim, 3, 1, 1), nn.LeakyReLU(0.2)]
        
        # Example: A few styled conv blocks or simple conv blocks
        # For simplicity, let's use ResBlks (non-styled, as style is injected via w later or this is simpler)
        # Or, use StyledConv2d if Wclues map itself should be heavily styled by w.
        # The paper says "Wclues combines Sseg(x) to produce the weather clues map Wclues(x,w), which is domain-specific."
        # This implies w should influence this decoder.
        
        # Let's use a simple styled approach:
        self.style_affine = nn.Linear(style_dim, hidden_dim) # To transform w
        
        conv_blocks = []
        current_dim = hidden_dim
        for _ in range(num_conv_blocks):
            # Using a simplified styled block here for demonstration
            # A full WeightDemodResBlk might be overkill if Wclues map is simple.
            # This is a placeholder; a more robust styled block might be needed.
            conv_blocks.append(nn.Conv2d(current_dim, current_dim, 3, 1, 1))
            conv_blocks.append(nn.LeakyReLU(0.2))
            # InstanceNorm could be added here too.
        self.conv_body = nn.Sequential(*conv_blocks)
        
        self.to_blending_map = nn.Sequential(
            nn.Conv2d(current_dim, 1, 3, 1, 1),
            nn.Sigmoid() # Output a map between 0 and 1
        )
        self.initial_conv = nn.Conv2d(sseg_map_channels, hidden_dim, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2)


    def forward(self, sseg_map, w_style):
        x = self.act(self.initial_conv(sseg_map))
        
        # Inject style w_style
        style_modulation = self.style_affine(w_style).unsqueeze(-1).unsqueeze(-1) # B x C x 1 x 1
        x = x * style_modulation # Simple multiplicative modulation for example
        
        x = self.conv_body(x)
        blending_map = self.to_blending_map(x)
        return blending_map

class GgloDecoder(nn.Module): # Renamed from WcluesDecoder in previous state
    def __init__(self, bottleneck_dim_in, style_dim, num_upsamples, encoder_skip_dims_reversed, target_final_dim):
        super().__init__()
        self.style_dim = style_dim
        
        self.decode_blocks = nn.ModuleList()
        self.skip_align_convs = nn.ModuleList()

        curr_dim = bottleneck_dim_in

        for i in range(num_upsamples):
            dim_out_block = max(curr_dim // 2, target_final_dim) if i < num_upsamples - 1 else target_final_dim
            dim_out_block = max(dim_out_block, style_dim)

            self.decode_blocks.append(
                WeightDemodResBlk(curr_dim, dim_out_block, style_dim, upsample=True)
            )
            
            if i < len(encoder_skip_dims_reversed): # Ensure we have a skip for this level
                current_encoder_skip_dim = encoder_skip_dims_reversed[i]
                self.skip_align_convs.append(
                    nn.Conv2d(current_encoder_skip_dim, dim_out_block, kernel_size=1, stride=1, padding=0)
                )
            else: # Should not happen if encoder_skip_dims_reversed is correctly sized
                self.skip_align_convs.append(None)


            curr_dim = dim_out_block
            
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(curr_dim, affine=True), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(curr_dim, 3, 1, 1, 0)
        )

    def forward(self, bottleneck_f, w_target, encoder_skip_features_reversed):
        x = bottleneck_f
        
        for i in range(len(self.decode_blocks)):
            x = self.decode_blocks[i](x, w_target)
            
            if i < len(encoder_skip_features_reversed) and self.skip_align_convs[i] is not None:
                skip_f_to_align = encoder_skip_features_reversed[i]
                
                if x.shape[2:] != skip_f_to_align.shape[2:]:
                    skip_f_to_align = F.interpolate(skip_f_to_align, size=x.shape[2:], mode='bilinear', align_corners=False)
                
                aligned_skip_f = self.skip_align_convs[i](skip_f_to_align)
                x = x + aligned_skip_f 
        
        return self.to_rgb(x)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=5, hidden_dim=512): # num_domains for WeatherGAN
        super().__init__()
        self.num_domains = num_domains
        layers = []
        layers += [nn.Linear(latent_dim, hidden_dim)]
        layers += [nn.ReLU()]
        for _ in range(3): # 3 hidden layers in shared part
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, style_dim)))

    def forward(self, z, y): # y is domain label
        h = self.shared(z)
        out = []
        for i in range(self.num_domains):
            out.append(self.unshared[i](h))
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        
        idx = torch.arange(y.size(0)).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=5, max_conv_dim=512): # num_domains for WeatherGAN
        super().__init__()
        self.num_domains = num_domains
        dim_in = 2**14 // img_size # Initial conv channels
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        # Downsampling blocks
        # Target 4x4 feature map before final conv, as in original StarGAN v2 StyleEncoder
        num_downsamples = int(np.log2(img_size / 4)) # e.g., 256/4=64 -> log2(64)=6 downsamplings

        for _ in range(num_downsamples):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, normalize=False, downsample=True)] # Original SE uses non-normalized ResBlks
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)] # Conv on 4x4 feature map
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains): # One linear layer per domain
            self.unshared.append(nn.Linear(dim_out, style_dim))

    def forward(self, x, y): # y is domain label
        h = self.shared(x)
        h = h.view(h.size(0), -1) # Flatten
        out = []
        for i in range(self.num_domains):
            out.append(self.unshared[i](h))
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        
        idx = torch.arange(y.size(0)).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s

class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=5, max_conv_dim=512): # num_domains for WeatherGAN
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        # Downsampling blocks, target 4x4 like StyleEncoder
        num_downsamples = int(np.log2(img_size / 4))
        
        for _ in range(num_downsamples):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, normalize=False, downsample=True)] # D uses non-normalized ResBlks
            dim_in = dim_out
        
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_in, dim_in, 4, 1, 0)] # Conv on 4x4
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_in, num_domains, 1, 1, 0)] # Output one logit per domain
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y): # y is domain label for which to get the logit
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        
        idx = torch.arange(x.size(0)).to(x.device)
        out = out[idx, y]  # (batch) -> Select logit for the given domain y
        return out

# Comment out or remove AdaIN related classes if fully replaced
# class AdaIN(nn.Module):
#     def __init__(self, style_dim, num_features):
#         super().__init__()
#         self.norm = nn.InstanceNorm2d(num_features, affine=False)
#         self.fc = nn.Linear(style_dim, num_features*2)

#     def forward(self, x, s):
#         h = self.fc(s)
#         h = h.view(h.size(0), h.size(1), 1, 1)
#         gamma, beta = torch.chunk(h, chunks=2, dim=1)
#         return (1 + gamma) * self.norm(x) + beta

# class AdainResBlk(nn.Module):
#   ... (definition as before) ...

# class AdaINWclues(nn.Module):
#   ... (definition as before) ...