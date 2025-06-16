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
from torchvision import transforms # Added for resizing

from core.model import Generator, MappingNetwork, StyleEncoder, Discriminator
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics


# Helper functions (adv_loss, r1_reg, VGGPerceptualLoss)
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


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device, layers=[20]): # Default layer index 20 for VGG16 features
        super().__init__()
        self.device = device
        self.selected = set(layers)
        # Only load up to the highest required layer to save GPU memory
        vgg_full = vgg16(pretrained=True)
        max_idx = max(self.selected)
        # Ensure max_idx is valid for vgg16.features
        # vgg16.features has 31 layers (0-30)
        if max_idx > 30 : max_idx = 30 
        
        all_layers = list(vgg_full.features[: max_idx + 1])
        # free the rest of the model
        del vgg_full
        # split layers into two contiguous parts for two-GPU assignment (if applicable)
        # Note: StarGANv2's original VGGPerceptualLoss had complex multi-GPU logic.
        # Simplifying for now, assuming single device or let DataParallel handle it if VGG is part of the main model.
        # For WeatherGAN, VGG is a loss, not part of the generator/discriminator directly moved to multiple GPUs.
        
        self.layers = nn.ModuleList()
        for layer in all_layers:
            layer.to(device).eval()
            for p in layer.parameters(): p.requires_grad = False
            self.layers.append(layer)
        
        # Store device for each layer if complex multi-GPU is needed later
        # self.devices = [device] * len(self.layers) 

    def forward(self, gen, real):
        # Images expected to be in [-1, 1] range, VGG expects [0,1] and normalized
        xg = (gen + 1) / 2 
        xr = (real + 1) / 2
        
        # Normalize using ImageNet mean and std would be more standard for VGG
        # However, StarGANv2's LPIPS and original perceptual loss might not do this explicitly here.
        # Sticking to (x+1)/2 for now as per common practice in GANs if VGG is used this way.

        feats_g, feats_r = [], []
        # Current device for all layers
        current_device = self.device 

        temp_xg = xg
        temp_xr = xr
        for idx, layer in enumerate(self.layers):
            temp_xg = layer(temp_xg.to(current_device))
            temp_xr = layer(temp_xr.to(current_device))
            if idx in self.selected:
                feats_g.append(temp_xg)
                feats_r.append(temp_xr)
        
        # L1 loss between selected features
        loss = sum(F.l1_loss(g, r) for g, r in zip(feats_g, feats_r))
        return loss


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

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        # EMA networks are not explicitly mentioned in WeatherGAN paper, so their setup is commented out.
        # If self.nets_ema is None (as per build_model if EMA is disabled),
        # subsequent code must use self.nets for evaluation/sampling.
        # for name, module in self.nets_ema.items():
        #     setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan' or net == 'percep' or net == 'ssim': # Do not create optimizers for FAN, VGG perceptual loss, or SSIM
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                # CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema), # EMA checkpointing disabled
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else: # For 'sample' or 'eval' mode
            # If EMA networks are not used/trained, load the main network checkpoints.
            # Original StarGANv2 loaded '{:06d}_nets_ema.ckpt' here.
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets)]

        self.to(self.device)
        
        if args.lambda_beta > 0 or args.lambda_cyc > 0 or args.lambda_inv > 0: # Perceptual loss needed for Lcyc or Linv
            self.percep = VGGPerceptualLoss(self.device)
            for p in self.percep.parameters(): p.requires_grad = False
            self.nets['percep'] = self.percep # Make it accessible
            
        # Add SSIM for Lcyc and Linv
        if args.lambda_cyc > 0 or args.lambda_inv > 0: # lambda_inv for Linv
            # from pytorch_msssim import SSIM # Moved import to top
            self.ssim = SSIM(data_range=1.0, size_average=True, channel=3) # Assuming images are normalized to [-1, 1] then scaled to [0,1] for SSIM
            self.nets['ssim'] = self.ssim # Make it accessible
            
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

        # Initialize loss functions
        self.mae_loss = torch.nn.L1Loss()
        self.celoss = torch.nn.CrossEntropyLoss() # Added for L_c (weather classification)
        self.seg_loss_fn = torch.nn.CrossEntropyLoss() # Added for L_s (segmentation)

        if self.args.distributed:
            # For distributed training, wrap the networks with DDP (DistributedDataParallel)
            for name, module in self.nets.items():
                if isinstance(module, nn.Module):
                    self.nets[name] = nn.parallel.DistributedDataParallel(module, device_ids=[self.device])

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
        # nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
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
            try:
                x_real, y_org, gt_seg_map_real = next(fetcher_train)
            except StopIteration:
                fetcher_train = iter(loaders.src)
                x_real, y_org, gt_seg_map_real = next(fetcher_train)
            
            # fetch reference images and labels
            try:
                x_ref, y_ref = next(fetcher_ref)
            except StopIteration:
                fetcher_ref = iter(loaders.ref)
                x_ref, y_ref = next(fetcher_ref)
            
            try:
                x_ref2, y_ref2 = next(fetcher_ref)
            except StopIteration:
                fetcher_ref = iter(loaders.ref) # re-initialize if needed, though typically ref loader is longer or wraps
                x_ref2, y_ref2 = next(fetcher_ref)


            x_real, y_org, gt_seg_map_real = x_real.to(self.device), y_org.to(self.device), gt_seg_map_real.to(self.device)
            x_ref, y_ref = x_ref.to(self.device), y_ref.to(self.device)
            x_ref2, y_ref2 = x_ref2.to(self.device), y_ref2.to(self.device)

            # train the discriminator
            with torch.amp.autocast('cuda'):
                d_loss, d_losses_latent = compute_d_loss(
                    nets, args, x_real, y_org, y_trg, x_fake=x_fake, device=self.device)
            self._reset_grad()
            scaler.scale(d_loss).backward()
            scaler.step(optims.discriminator)
            scaler.update()

            with torch.amp.autocast('cuda'):
                d_loss, d_losses_ref = compute_d_loss(
                    nets, args, x_real, y_org, y_trg, x_fake=x_fake, device=self.device)
            self._reset_grad()
            scaler.scale(d_loss).backward()
            scaler.step(optims.discriminator)
            scaler.update()

            # train the generator
            # latent-guided image synthesis
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg2 = nets.mapping_network(z_trg2, y_trg) # For L_wd (diversity loss)
            
            # WeatherGAN generator outputs: x_fake, s_trg_style_reconst, s_seg_map, s_seg_cls_logits, w_clues_map, g_glo_out
            x_fake, s_trg_style_reconst, s_seg_map, s_seg_cls_logits, w_clues_map, g_glo_out = nets.generator(x_real, s_trg)
            
            g_loss_latent, g_losses_latent = self.compute_g_loss(
                nets, args, x_real, y_org, y_trg, 
                s_trg=s_trg, s_trg2=s_trg2, # Pass s_trg2 for diversity
                x_fake=x_fake, s_trg_style_reconst=s_trg_style_reconst,
                s_seg_map=s_seg_map, s_seg_cls_logits=s_seg_cls_logits,
                g_glo_out=g_glo_out, gt_seg_map_real=gt_seg_map_real,
                device=self.device
            )
            self._reset_grad()
            scaler.scale(g_loss_latent).backward()
            scaler.step(optims.generator)
            scaler.step(optims.mapping_network)
            scaler.step(optims.style_encoder)
            scaler.update()

            with torch.amp.autocast('cuda'):
                g_loss, g_losses_ref = compute_g_loss(
                    nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
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
            
            # WeatherGAN uses -lambda_wd * Lwd, StarGANv2 uses -lambda_ds * Lds. These are equivalent if Lwd = Lds.
            # WeatherGAN also has Lseg and Lwrc. These need to be added to g_loss calculation.

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
                # Use self.nets for debugging if self.nets_ema is not being used/trained
                utils.debug_image(self.nets if self.nets_ema is None else self.nets_ema, args, inputs=inputs_val, step=i+1)

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
                # Use self.nets for evaluation if self.nets_ema is not being used/trained
                calculate_metrics(self.nets if self.nets_ema is None else self.nets_ema, args, i+1, mode='latent')
                calculate_metrics(self.nets if self.nets_ema is None else self.nets_ema, args, i+1, mode='reference')

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        # Use self.nets for sampling if self.nets_ema is not being used/trained
        nets_to_use = self.nets if self.nets_ema is None else self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter) # This will load _nets.ckpt due to __init__ changes

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_to_use, args, src.x, ref.x, ref.y, fname)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_to_use, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        # Use self.nets for evaluation if self.nets_ema is not being used/trained
        nets_to_use = self.nets if self.nets_ema is None else self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter) # This will load _nets.ckpt due to __init__ changes
        calculate_metrics(nets_to_use, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_to_use, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, x_real, y_org, y_trg, x_fake=None, z_trg=None, x_ref=None, device='cuda'):
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

        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, 
                   s_trg, s_trg2, x_fake, s_trg_style_reconst,
                   s_seg_map, s_seg_cls_logits, g_glo_out,
                   gt_seg_map_real, x_ref=None, device='cuda'):
    # s_trg is the target style for x_fake generation
    # s_trg2 is another target style for diversity loss L_wd
    
    # adversarial loss
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss (L_inv for WeatherGAN)
    # s_trg_style_reconst is E(G(x, s_trg))
    loss_inv = self.mae_loss(s_trg_style_reconst, s_trg)
    
    # weather diversity loss (L_wd for WeatherGAN)
    # Generate another fake image with s_trg2
    x_fake2, _, _, _, _, _ = nets.generator(x_real, s_trg2)
    loss_wd = self.mae_loss(x_fake, x_fake2)

    # cycle-consistency loss (L_cyc for WeatherGAN)
    s_org = nets.style_encoder(x_real, y_org) # w_src
    x_recon, _, _, _, _, _ = nets.generator(x_fake, s_org)
    loss_cyc = self.mae_loss(x_recon, x_real) # L1 cycle loss

    # segmentation loss (L_seg for WeatherGAN = L_c + L_s)
    # L_c: weather classification loss for input x_real based on S_seg branch
    # s_seg_cls_logits is from S_seg(x_real_features, s_trg)
    # Target for L_c should be y_org (weather of x_real)
    loss_c = self.celoss(s_seg_cls_logits, y_org)

    # L_s: semantic segmentation loss for input x_real based on S_seg branch
    # s_seg_map is from S_seg(x_real_features, s_trg)
    # Resize gt_seg_map_real to match s_seg_map spatial dimensions
    s_seg_map_size = s_seg_map.shape[2:] # H', W'
    # Assuming gt_seg_map_real is B x H_orig x W_orig (long tensor)
    # Need to ensure gt_seg_map_real is [B, H, W] and then unsqueeze to [B, 1, H, W] for resize
    if gt_seg_map_real.ndim == 3: # B, H, W
        gt_seg_map_real_unsqueezed = gt_seg_map_real.unsqueeze(1).float()
    elif gt_seg_map_real.ndim == 4: # B, 1, H, W
        gt_seg_map_real_unsqueezed = gt_seg_map_real.float()
    else: # Should not happen if data loader is correct
        raise ValueError(f"gt_seg_map_real has unexpected ndim: {gt_seg_map_real.ndim}")

    gt_seg_map_resized = transforms.Resize(s_seg_map_size, interpolation=transforms.InterpolationMode.NEAREST)(gt_seg_map_real_unsqueezed)
    gt_seg_map_resized = gt_seg_map_resized.squeeze(1).long() # Back to B x H' x W'
    
    loss_s = self.seg_loss_fn(s_seg_map, gt_seg_map_resized)
    loss_seg = loss_c + loss_s

    # weather-irrelevant content reconstruction loss (L_wrc for WeatherGAN)
    # g_glo_out is G_glo(x_real, s_trg)
    x_real_hp = nets.generator.high_pass(x_real)
    loss_wrc = self.mae_loss(g_glo_out, x_real_hp)

    loss_wing = torch.zeros(1).to(device)
    if args.lambda_wing > 0:
        if x_ref is None: # wing loss is only for reference-guided if x_ref is given
             # For latent-guided, x_real is used as reference for wing loss in original StarGANv2
             x_ref_wing = x_real 
        else:
             x_ref_wing = x_ref

        # Original StarGANv2 wing loss calculation:
        # x_fake_face_landmarks = nets.fan(nets.generator.get_shared_feature(x_fake)) # FAN needs features from G's encoder part
        # x_ref_face_landmarks = nets.fan(nets.generator.get_shared_feature(x_ref_wing)) # Or precomputed landmarks
        # For simplicity, assuming FAN takes images directly or landmarks are precomputed
        # This part needs careful adaptation if wing loss is to be used with WeatherGAN structure.
        # The paper for WeatherGAN does not mention wing loss.
        # Placeholder:
        # loss_wing = self.mae_loss(x_fake_face_landmarks, x_ref_face_landmarks)
        pass # Skipping wing loss for now as it's complex to integrate and not in WeatherGAN


    loss = args.lambda_adv * loss_adv + \
           args.lambda_inv * loss_inv + \
           args.lambda_wd * loss_wd + \
           args.lambda_cyc * loss_cyc + \
           args.lambda_seg * loss_seg + \
           args.lambda_wrc * loss_wrc + \
           args.lambda_wing * loss_wing # lambda_wing is likely 0 for WeatherGAN

    return loss, Munch(adv=loss_adv.item(),
                       inv=loss_inv.item(), # Was sty
                       wd=loss_wd.item(),   # Was ds
                       cyc=loss_cyc.item(),
                       seg=loss_seg.item(), # New
                       wrc=loss_wrc.item(), # New
                       wing=loss_wing.item())


def compute_d_loss(nets, args, x_real, y_org, y_trg, x_fake=None, z_trg=None, x_ref=None, device='cuda'):
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

        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())