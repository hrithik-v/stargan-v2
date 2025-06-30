"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
# import os
# os.environ["WANDB_MODE"] = "disabled"

import copy
import math

from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F  # ensure functional alias is present
from torch.nn.utils import spectral_norm
from core.wing import FAN


# Add weight demodulation modulated convolution class
class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, style_dim, demodulate=True):
        super().__init__()
        self.eps = 1e-8
        self.demodulate = demodulate
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel_size, kernel_size))
        self.modulation = nn.Linear(style_dim, in_ch)
        nn.init.kaiming_normal_(self.weight, a=0.2, mode='fan_in')
        
    def forward(self, x, style):
        b, c, h, w = x.shape
        style = self.modulation(style).view(b, 1, c, 1, 1) + 1  # [-1,1] -> [0,2]
        
        # Modulate weights
        weight = self.weight * style
        if self.demodulate:
            d = torch.rsqrt(weight.pow(2).sum([2,3,4]) + self.eps)
            weight = weight * d.view(b, -1, 1, 1, 1)
        
        # Grouped convolution
        x = x.view(1, b*c, h, w)
        weight = weight.view(b*self.weight.size(1), *self.weight.shape[2:])
        out = F.conv2d(x, weight, padding=self.weight.size(-1)//2, groups=b)
        return out.view(b, -1, h, w)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False, use_spectral_norm=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, use_spectral_norm)

    def _build_weights(self, dim_in, dim_out, use_spectral_norm):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            if self.learned_sc:
                self.conv1x1 = spectral_norm(self.conv1x1)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


# Custom skip connection that ignores style
class Skip(nn.Module):
    def forward(self, x, s=None):
        return x


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.actv = nn.LeakyReLU(0.2)
        
        # Main path
        self.conv1 = ModulatedConv2d(dim_in, dim_out, 3, style_dim)
        self.conv2 = ModulatedConv2d(dim_out, dim_out, 3, style_dim)
        
        # Skip connection
        if dim_in != dim_out or upsample:
            self.skip = ModulatedConv2d(dim_in, dim_out, 1, style_dim)
        else:
            self.skip = Skip()

    def forward(self, x, s):
        residual = x
        if self.upsample:
            residual = F.interpolate(residual, scale_factor=2, mode='bilinear')
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
        
        x = self.conv1(self.actv(x), s)
        x = self.conv2(self.actv(x), s)
        
        skip = self.skip(residual, s)
        return (x + skip) / math.sqrt(2)
    
class HighPass(nn.Module):
    def __init__(self, w_hpf):
        super().__init__()
        self.register_buffer('kernel', 
                            torch.tensor([[-1, -1, -1],
                                          [-1, 8, -1],
                                          [-1, -1, -1]]) / w_hpf)
    
    def forward(self, x):
        kernel = self.kernel.expand(x.size(1), 1, 3, 3)
        return F.conv2d(x, kernel, padding=1, groups=x.size(1))



class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, 
                 num_domains=5, seg_classes=5, w_hpf=1):
        """
        WeatherGAN Generator
        Args:
            img_size: Input image size (assumed square)
            style_dim: Dimension of style vector
            max_conv_dim: Maximum channels in convolutional layers
            num_domains: Number of weather domains (classes)
            seg_classes: Number of segmentation classes
            w_hpf: High-pass filter weight (0 to disable)
        """
        super().__init__()
        # Initial channel dimension (scales with image size)
        dim_in = 2**14 // img_size
        
        # 1. Shared Encoder
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encoder = nn.ModuleList()
        
        # 2. Weather Segmentation Module (Domain-Invariant)
        self.seg_decoder = nn.ModuleList()
        self.to_seg = nn.Sequential(
            nn.Conv2d(dim_in, seg_classes, 1),
            nn.Softmax(dim=1)
        )

        # 3. Weather Clues Module (Domain-Specific)
        self.clue_processor = ClueProcessor(seg_classes, style_dim)

        # 4. Global Weather Translation Module (Domain-Specific)
        self.global_decoder = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1),
            nn.Tanh()
        )
        
        # Build encoder/decoder blocks
        repeat_num = int(math.log2(img_size)) - 2
        channels = dim_in
        
        # Encoder and decoder blocks
        for _ in range(repeat_num):
            dim_out = min(channels * 2, max_conv_dim)
            
            # Encoder block (downsample)
            self.encoder.append(
                ResBlk(channels, dim_out, normalize=True, downsample=True)
            )
            
            # Segmentation decoder block (upsample, no style)
            self.seg_decoder.insert(0, nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(dim_out, channels, 3, 1, 1),
                nn.InstanceNorm2d(channels, affine=True),
                nn.LeakyReLU(0.2)
            ))
            
            # Global decoder block (upsample with style)
            self.global_decoder.insert(0,
                AdainResBlk(dim_out, channels, style_dim, upsample=True)
            )
            
            channels = dim_out
        
        # Bottleneck blocks
        for _ in range(1):
            self.encoder.append(
                ResBlk(channels, channels, normalize=True)
            )
            self.seg_decoder.insert(0,
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, 1, 1),
                    nn.InstanceNorm2d(channels, affine=True),
                    nn.LeakyReLU(0.2)
                )
            )

            self.global_decoder.insert(0,
                AdainResBlk(channels, channels, style_dim)
            )
        # 5. Weather Classifier (must be after channels is finalized)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_domains)
        )

        # High-pass filter (optional)
        self.hpf = HighPass(w_hpf) if w_hpf > 0 else None

    def forward(self, x, style_code, masks=None):
        """
        Forward pass through WeatherGAN generator
        Args:
            x: Input image tensor (B, 3, H, W)
            style_code: Style vector (B, style_dim)
        Returns:
            output: Translated image
            seg_map: Weather segmentation map
            clues_mask: Weather clues mask
            weather_logits: Weather classification logits
        """
        # --- Shared Encoder ---
        h = self.from_rgb(x)
        for block in self.encoder:
            h = block(h)
        
        # --- Weather Classification ---
        weather_logits = self.classifier(h)
        
        # --- Weather Segmentation (Domain-Invariant) ---
        seg_feat = h
        for block in self.seg_decoder:
            seg_feat = block(seg_feat)
        seg_map = self.to_seg(seg_feat)
        
        # --- Weather Clues Processing (Domain-Specific) ---
        # Detach segmentation to preserve domain-invariance
        clues_input = seg_map.detach()
        clues_mask = self.clue_processor(clues_input, style_code)
        
        # --- Global Weather Translation (Domain-Specific) ---
        glo_feat = h
        for block in self.global_decoder:
            glo_feat = block(glo_feat, style_code)
        glo_out = self.to_rgb(glo_feat)
        
        # Apply high-pass filter if enabled
        # print(f"glo_out shape: {glo_out.shape}, Max: {glo_out.max().item()}, Min: {glo_out.min().item()}, Mean: {glo_out.mean().item()}")
        # if self.hpf:
        #     glo_out = self.hpf(glo_out)
        
        # --- Final Composition ---
        # Preserve weather-invariant regions (no gradients to original image)

        # print(f"clues_mask shape: {clues_mask.shape}, Max: {clues_mask.max().item()}, Min: {clues_mask.min().item()}, Mean: {clues_mask.mean().item()}")
        # print(f"glo_out shape: {glo_out.shape}, Max: {glo_out.max().item()}, Min: {glo_out.min().item()}, Mean: {glo_out.mean().item()}")

        
        # clues_mask = clues_mask*2
        with torch.no_grad():
            preserved = x * (1 - clues_mask)
        
        # Combine with weather-translated regions
        translated = glo_out * clues_mask
        output = preserved + translated
        
        return output, seg_map, weather_logits, clues_mask
        # return {
        #     'image': output,
        #     'seg': seg_map,
        #     'weather_logits': weather_logits,
        #     'clues': clues_mask
        # }



class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y, p=1.0):
        """
        Generate weather control code with intensity interpolation.
        z: random latent code (batch, latent_dim)
        y: target domain labels (batch,)
        p: intensity factor for interpolation
        """
        h = self.shared(z)
      #model print(f"h shape: {h.shape}, Max: {h.max().item()}, Min: {h.min().item()}")
        styles = []
        for layer in self.unshared:
            styles.append(layer(h))
        styles = torch.stack(styles, dim=1)  # (batch, num_domains, style_dim)
        # Compute domain-invariant mean
        mean_style = styles.mean(dim=1)  # (batch, style_dim)
        # Select domain-specific style
        idx = torch.arange(y.size(0), device=y.device)
        w_i = styles[idx, y]  # (batch, style_dim)
        # Interpolate with intensity p
        w = mean_style + p * (w_i - mean_style)
      #model print(f"w shape: {w.shape}, Max: {w.max().item()}, Min: {w.min().item()}")
        return w


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, normalize=True, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
      #model print(f'StyleEncoder input shape: {x.shape}, Max: {x.max().item()}, Min: {x.min().item()}')
        h = self.shared(x)
      #model print(f'StyleEncoder shared output shape: {h.shape}, Max: {h.max().item()}, Min: {h.min().item()}')
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
      #model print(f'StyleEncoder output shape: {s.shape}, Max: {s.max().item()}, Min: {s.min().item()}')
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(3, dim_in, 3, 1, 1))]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True, use_spectral_norm=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 4, 1, 0))]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, num_domains, 1, 1, 0))]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
      #model print(f'Discriminator output shape: {out.shape}, Max: {out.max().item()}, Min: {out.min().item()}')
        out = out.view(out.size(0), -1)  # (batch, num_domains)

        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out




# Custom module for processing clues with style conditioning
class ClueProcessor(nn.Module):
    def __init__(self, in_ch, style_dim):
        super().__init__()
        self.conv1 = ModulatedConv2d(in_ch, 64, 3, style_dim)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = ModulatedConv2d(64, 64, 3, style_dim)
        self.act2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, s):
        x = self.conv1(x, s)
        x = self.act1(x)
        x = self.conv2(x, s)
        x = self.act2(x)
        x = self.conv3(x)
        return self.sigmoid(x)


def build_model(args):
    # Build generator with segmentation classes
    generator = nn.DataParallel(Generator(
        args.img_size, args.style_dim, args.max_conv_dim,
        args.num_domains, args.seg_classes, args.w_hpf))
    mapping_network = nn.DataParallel(MappingNetwork(args.latent_dim, args.style_dim, args.num_domains))
    style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains, args.max_conv_dim))
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains, args.max_conv_dim))
    # generator_ema = copy.deepcopy(generator)
    # mapping_network_ema = copy.deepcopy(mapping_network)
    # style_encoder_ema = copy.deepcopy(style_encoder)

    # print(generator)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = None
    # nets_ema = Munch(generator=generator_ema,
    #                  mapping_network=mapping_network_ema,
    #                  style_encoder=style_encoder_ema)

    # if args.w_hpf > 0:
    #     fan = nn.DataParallel(FAN(fname_pretrained=args.wing_path).eval())
    #     fan.get_heatmap = fan.module.get_heatmap
    #     nets.fan = fan
    #     nets_ema.fan = fan

    return nets, nets_ema
