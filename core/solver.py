"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb  # We use wandb for logging
from torchvision.models import vgg16  # for perceptual loss

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics


# inline SSIM implementation for structural consistency losses
def ssim(img1, img2, window_size=3, size_average=True):
    """Compute structural similarity index between img1 and img2."""
    # constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    # mean
    mu1 = F.avg_pool2d(img1, window_size, 1, window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, 1, window_size//2)
    # variances and covariance
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, window_size//2) - mu1 * mu1
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, window_size//2) - mu2 * mu2
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size//2) - mu1 * mu2
    # SSIM map
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize wandb
        wandb.login(key=args.wandb_api_token)
        wandb.init(project="starganv2", name=args.wandb_name,
        id=args.wandb_id, 
        resume= "allow" if args.wandb_resume else None, 
        config=vars(args)
        )

        # build core networks
        self.nets, self.nets_ema = build_model(args)
        # register utils and eval networks
        # attach inline SSIM to nets for structural consistency
        if args.lambda_cyc > 0 or args.lambda_inv > 0:
            self.nets['ssim'] = ssim  # use inline SSIM function

        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # for name, module in self.nets_ema.items():
        #     setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                # Skip nets that do not have parameters (e.g., ssim)
                if not hasattr(self.nets[net], 'parameters'):
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                # CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        
        if args.lambda_beta > 0:
            # Initialize a single VGGPerceptualLoss instance for perceptual loss
            self.percep = VGGPerceptualLoss(self.device)
            # Ensure no gradients are computed for perceptual network
            for p in self.percep.parameters(): p.requires_grad = False
            # DataParallelize VGG perceptual loss across available GPUs
            self.percep = nn.DataParallel(self.percep)
            # Expose in nets for compute_g_loss lookup
            self.nets['percep'] = self.percep
            
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema  # use EMA nets for eval
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, loaders.seg, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()

        scaler = torch.amp.GradScaler('cuda')  # for mixed precision

        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            masks = None # nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # train the discriminator
            with torch.amp.autocast('cuda'):
                d_loss, d_losses_latent = compute_d_loss(
                    nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
            self._reset_grad()
            scaler.scale(d_loss).backward()
            scaler.step(optims.discriminator)
            scaler.update()

            with torch.amp.autocast('cuda'):
                d_loss, d_losses_ref = compute_d_loss(
                    nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
            self._reset_grad()
            scaler.scale(d_loss).backward()
            scaler.step(optims.discriminator)
            scaler.update()

            # train the generator
            with torch.amp.autocast('cuda'):
                # include segmentation ground-truth masks
                seg_gt = inputs.get('seg_gt', None)
                g_loss, g_losses_latent = compute_g_loss(
                    nets, args, x_real, y_org, y_trg,
                    z_trgs=[z_trg, z_trg2], masks=masks, seg_gt=seg_gt)
            self._reset_grad()
            scaler.scale(g_loss).backward()
            scaler.step(optims.generator)
            scaler.step(optims.mapping_network)
            scaler.step(optims.style_encoder)
            scaler.update()

            with torch.amp.autocast('cuda'):
                # include segmentation masks for reference mode if available
                seg_gt = inputs.get('seg_gt', None)
                g_loss, g_losses_ref = compute_g_loss(
                    nets, args, x_real, y_org, y_trg,
                    x_refs=[x_ref, x_ref2], masks=masks, seg_gt=seg_gt)
            self._reset_grad()
            scaler.scale(g_loss).backward()
            scaler.step(optims.generator)
            scaler.update()

            # compute moving average of network parameters
            # moving_average(nets.generator, nets_ema.generator, beta=0.999)
            # moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            # moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            print('\rIteration [%i/%i]' % (i+1, args.total_iters), end='')

            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                # Log metrics to wandb
                wandb.log({**all_losses, 'iteration': i+1})

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # Save latest checkpoints every 100 iterations
            if (i+1) % 100 == 0:
                latest_ckpt_nets = CheckpointIO(ospj(args.checkpoint_dir, 'latest_nets.ckpt'), data_parallel=True, **self.nets)
                latest_ckpt_optims = CheckpointIO(ospj(args.checkpoint_dir, 'latest_optims.ckpt'), **self.optims)
                latest_ckpt_nets.save(step=i+1)
                latest_ckpt_optims.save(step=i+1)

            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i+1, mode='latent')
                calculate_metrics(nets_ema, args, i+1, mode='reference')

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake, _seg, _cls = nets.generator(x_real, s_trg, masks=masks)

    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())



def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None, seg_gt=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    # forward generator
    x_fake, seg_pred, cls_logits = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2, _seg2, _cls2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle structural perceptual consistency (Eq.7): SSIM + perceptual VGG loss
    masks = None # nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec, _seg_rec, _cls_rec = nets.generator(x_fake, s_org, masks=masks)
    # SSIM term between x and reconstructed image
    loss_cyc_ssim = 1 - nets.ssim(x_real, x_rec)
    # perceptual VGG term
    loss_cyc_per = nets.percep(x_rec, x_real) if args.lambda_beta > 0 else 0
    # total cycle loss
    loss_cyc = loss_cyc_ssim + loss_cyc_per

    # weather-invariant consistency loss
    loss_inv_ssim = 1 - nets.ssim(x_real, x_fake)
    loss_inv_per = nets.percep(x_fake, x_real) if args.lambda_beta > 0 else 0
    loss_inv = loss_inv_ssim + loss_inv_per

    # segmentation and classification losses
    # weakly-supervised segmentation & classification losses (Eq.5)
    if seg_gt is not None and args.lambda_seg > 0:
        # multi-class segmentation: seg_pred is raw logits (N, num_domains, H, W), seg_gt has shape (N, H, W)
        loss_s = F.cross_entropy(seg_pred, seg_gt)
        loss_c = F.cross_entropy(cls_logits, y_org)
        loss_seg = loss_s + loss_c
    else:
        loss_seg = 0

    # total generator loss
    loss = loss_adv \
        + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds \
        + args.lambda_cyc * loss_cyc \
        + args.lambda_inv * loss_inv \
        + args.lambda_seg * loss_seg

    return loss, Munch(
         adv=loss_adv.item(),
         sty=loss_sty.item(),
         ds=loss_ds.item(),
         cyc=loss_cyc.item(),
         inv=loss_inv.item(),
         seg=loss_seg.item(),
         cls=loss_c.item()
     )


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


# Add VGGPerceptualLoss implementation for perceptual loss
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device, layers=[20]):
        super().__init__()
        self.device = device
        self.selected = set(layers)
        # Only load up to the highest required layer to save GPU memory
        vgg_full = vgg16(pretrained=True)
        max_idx = max(self.selected)
        all_layers = list(vgg_full.features[: max_idx + 1])
        # free the rest of the model
        del vgg_full
        # split layers into two contiguous parts for two-GPU assignment
        if torch.cuda.device_count() > 1:
            dev1 = torch.device('cuda:0')
            dev0 = torch.device('cuda:1')
        else:
            dev0 = device
            dev1 = device
        split_idx = len(all_layers) // 2
        self.layers = nn.ModuleList()
        self.devices = []
        for idx, layer in enumerate(all_layers):
            # first half to dev0, second half to dev1
            dev = dev0 if idx < split_idx else dev1
            layer.to(dev).eval()
            for p in layer.parameters(): p.requires_grad = False
            self.layers.append(layer)
            self.devices.append(dev)

    def forward(self, gen, real):
        xg = (gen + 1) / 2
        xr = (real + 1) / 2
        feats_g, feats_r = [], []
        for idx, layer in enumerate(self.layers):
            dev = self.devices[idx % len(self.devices)]
            xg = layer(xg.to(dev))
            xr = layer(xr.to(dev))
            if idx in self.selected:
                feats_g.append(xg.to(self.device))
                feats_r.append(xr.to(self.device))
        loss = sum(F.l1_loss(g, r) for g, r in zip(feats_g, feats_r))
        return loss