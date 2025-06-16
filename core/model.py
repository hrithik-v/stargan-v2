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
from torch import nn
import torch.nn.functional as F  # ensure functional alias is present

from core.wing import FAN


# Add weight demodulation modulated convolution class
class ModulatedConv2d(nn.Module):
    """
    Conv2d layer with style-based weight modulation and demodulation (weight demodulation).
    """
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.demodulate = demodulate
        # weight shape: [1, out_channel, in_channel, k, k]
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        # mapping from style vector to modulation scale
        self.modulation = nn.Linear(style_dim, in_channel)
        # store channel dims
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

    def forward(self, x, style):
        batch, in_c, h, w = x.shape
        # compute style modulation
        style = self.modulation(style).view(batch, 1, in_c, 1, 1)
        # modulate
        weight = self.weight * style
        # demodulation
        if self.demodulate:
            demod = torch.rsqrt((weight * weight).sum([2,3,4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        # reshape for grouped convolution
        weight = weight.view(batch * self.out_channel, in_c, self.kernel_size, self.kernel_size)
        x = x.view(1, batch * in_c, h, w)
        out = F.conv2d(x, weight, padding=self.kernel_size//2, groups=batch)
        return out.view(batch, self.out_channel, h, w)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

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


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        # Use modulated convolution with weight demodulation
        self.conv1 = ModulatedConv2d(dim_in, dim_out, 3, style_dim)
        self.conv2 = ModulatedConv2d(dim_out, dim_out, 3, style_dim)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        # apply activation and upsample before modulated conv
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x, s)
        x = self.actv(x)
        x = self.conv2(x, s)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        return (out + self._shortcut(x)) / math.sqrt(2)


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        # Shared encoder
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        # Decode blocks for clues and global branches
        self.decode_clues = nn.ModuleList()
        self.decode_glo = nn.ModuleList()
        # Segmentation head
        self.to_seg = nn.Sequential(nn.Conv2d(dim_in, 1, 1), nn.Sigmoid())
        # Weather clues head
        self.to_clues = nn.Sequential(nn.InstanceNorm2d(dim_in, affine=True),
                                     nn.LeakyReLU(0.2),
                                     nn.Conv2d(dim_in, 1, 1),
                                     nn.Sigmoid())
        # Global translation head
        self.to_glo = nn.Sequential(nn.InstanceNorm2d(dim_in, affine=True),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(dim_in, 3, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            # shared decode for clues and glo
            self.decode_clues.insert(0, AdainResBlk(dim_out, dim_in, style_dim,
                                 w_hpf=w_hpf, upsample=True))
            self.decode_glo.insert(0, AdainResBlk(dim_out, dim_in, style_dim,
                                 w_hpf=w_hpf, upsample=True))
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode_clues.insert(0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))
            self.decode_glo.insert(0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None, p=1.0):
        # Shared encoding
        feat = self.from_rgb(x)
        for block in self.encode:
            feat = block(feat)
        # Segmentation map
        seg = self.to_seg(feat)
        # Weather clues branch
        clues_feat = feat
        for block in self.decode_clues:
            clues_feat = block(clues_feat, s)
        clues = self.to_clues(clues_feat)
        # Global translation branch
        glo_feat = feat
        for block in self.decode_glo:
            glo_feat = block(glo_feat, s)
        glo = self.to_glo(glo_feat)
        # Combine according to Eq. (1)
        out = clues * glo + (1 - clues) * x
        return out, seg


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
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


def build_model(args):
    generator = nn.DataParallel(Generator(args.img_size, args.style_dim, args.max_conv_dim, args.w_hpf))
    mapping_network = nn.DataParallel(MappingNetwork(args.latent_dim, args.style_dim, args.num_domains))
    style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains, args.max_conv_dim))
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains, args.max_conv_dim))
    # generator_ema = copy.deepcopy(generator)
    # mapping_network_ema = copy.deepcopy(mapping_network)
    # style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = None
    # nets_ema = Munch(generator=generator_ema,
    #                  mapping_network=mapping_network_ema,
    #                  style_encoder=style_encoder_ema)

    if args.w_hpf > 0:
        fan = nn.DataParallel(FAN(fname_pretrained=args.wing_path).eval())
        fan.get_heatmap = fan.module.get_heatmap
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema
