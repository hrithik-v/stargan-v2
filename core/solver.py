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
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics
import wandb


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # mixed-precision scaler
        self.scaler = GradScaler()

        self.nets, self.nets_ema = build_model(args)
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
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                # CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]

            # initialize wandb
            wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
            # watch networks for logging gradients and parameters
            # for name, net in self.nets.items():
            #     if name in ['generator', 'discriminator']:  # Specify networks to watch
            #         wandb.watch(net, log="all", log_freq=args.print_every)
                # To disable entirely, comment out the wandb.watch line:
                # pass
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        # move to device and use channels_last memory format for conv performance
        self.to(self.device, memory_format=torch.channels_last)
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
        nets_ema = self.nets_ema
        optims = self.optims

        # Fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        initial_lambda_ds = args.lambda_ds
        print('Start training...')
        start_time = time.time()

        # Determine iterations
        iters_per_epoch = len(fetcher.loader)
        num_epochs = args.num_epochs
        start_epoch = args.resume_epoch if args.resume_epoch > 0 else 0

        for epoch in range(start_epoch, num_epochs):
            pbar = tqdm(total=iters_per_epoch, initial=0, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

            for batch_idx in range(iters_per_epoch):

                # Fetch batch
                inputs = next(fetcher)
                x_real, y_org = inputs.x_src, inputs.y_src
                x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
                z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2
                masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

                # Discriminator step
                with autocast():
                    d_loss, d_losses_latent = compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
                self._reset_grad()
                self.scaler.scale(d_loss).backward()
                self.scaler.step(optims.discriminator)
                self.scaler.update()

                # Generator step
                with autocast():
                    g_loss, g_losses_latent = compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
                self._reset_grad()
                self.scaler.scale(g_loss).backward()
                self.scaler.step(optims.generator)
                self.scaler.step(optims.mapping_network)
                self.scaler.step(optims.style_encoder)
                self.scaler.update()

                # Collect and display losses
                all_losses = {}
                for loss, prefix in zip([d_losses_latent, g_losses_latent], ['D/latent_', 'G/latent_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds

                pbar.set_postfix({k: f"{v:.4f}" for k, v in all_losses.items()})
                pbar.update(1)

                # EMA update and lambda decay
                if args.ema:
                    moving_average(nets.generator, nets_ema.generator, beta=0.999)
                    moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
                    moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)
                if args.lambda_ds > 0:
                    args.lambda_ds -= (initial_lambda_ds / args.ds_epoch)

            # Logging, saving, evaluating
            if epoch % args.wandb_log == 0:
                wandb.log(all_losses, step=epoch)
            if epoch % args.save_every == 0:
                self._save_checkpoint(step=epoch)
            if epoch % args.eval_every == 0:
                calculate_metrics(nets, args, epoch, mode='latent')
                calculate_metrics(nets, args, epoch, mode='reference')

            pbar.close()  # Close tqdm at the end of each epoch


    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_epoch)

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
        nets = self.nets
        resume_epoch = args.resume_epoch
        self._load_checkpoint(args.resume_epoch)
        calculate_metrics(nets, args, step=resume_epoch, mode='latent')
        calculate_metrics(nets, args, step=resume_epoch, mode='reference')


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    # loss_reg = r1_reg(out, x_real)
    loss_reg = 0

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
                       reg=0)
                    #    reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    # assert (z_trgs is None) != (x_refs is None)
    # if z_trgs is not None:
    #     z_trg, z_trg2 = z_trgs
    # if x_refs is not None:
    #     x_ref, x_ref2 = x_refs
    z_trg, z_trg2 = z_trgs
    x_ref, x_ref2 = x_refs

    # adversarial loss
    # if z_trgs is not None:
    #     s_trg = nets.mapping_network(z_trg, y_trg)
    # else:
    #     s_trg = nets.style_encoder(x_ref, y_trg)
    s_trg = nets.mapping_network(z_trg, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)

    s_ref = nets.style_encoder(x_ref, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_ref))
    # loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    # if z_trgs is not None:
        # s_trg2 = nets.mapping_network(z_trg2, y_trg)
    # else:
        # s_trg2 = nets.style_encoder(x_ref2, y_trg)
    s_trg2 = nets.mapping_network(z_trg2, y_trg)
    
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())


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
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg