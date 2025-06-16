"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch

from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    print(args)
    # cudnn.benchmark = True``
    torch.manual_seed(args.seed)

    solver = Solver(args)

    if args.mode == 'train':
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.val_img_dir)) == args.num_domains
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers, max_per_class=args.max_per_class),
                        ref=get_train_loader(root=args.train_img_dir,
                                             which='reference',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers, max_per_class=args.max_per_class),
                        val=get_test_loader(root=args.val_img_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers))
        solver.train(loaders)
    elif args.mode == 'sample':
        assert len(subdirs(args.src_dir)) == args.num_domains
        assert len(subdirs(args.ref_dir)) == args.num_domains
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers),
                        ref=get_test_loader(root=args.ref_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers))
        solver.sample(loaders)
    elif args.mode == 'eval':
        solver.evaluate()
    elif args.mode == 'align':
        from core.wing import align_faces
        align_faces(args, args.inp_dir, args.out_dir)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=5,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')
    parser.add_argument('--max_conv_dim', type=int, default=512,
                        help='Maximum number of convolutional filters in generator and discriminator')
    # Add num_seg_classes for WeatherGAN S_seg branch
    parser.add_argument('--num_seg_classes', type=int, default=5, # Default to 5 as per WeatherGAN paper (clear, overcast, rain, snow, fog)
                        help='Number of segmentation classes for S_seg branch')


    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    # WeatherGAN lambda_cyc for Lcyc = (1-SSIM) + VGG. Original StarGANv2 lambda_cyc was for L1.
    parser.add_argument('--lambda_cyc', type=float, default=1.0, # WeatherGAN paper: 1.0
                        help='Weight for cycle structural perceptual consistency loss (Lcyc)')
    # lambda_sty in StarGANv2 maps to lambda_inv in WeatherGAN (weather-invariant consistency loss)
    parser.add_argument('--lambda_inv', type=float, default=1.0, # WeatherGAN paper: 1.0
                        help='Weight for weather-invariant consistency loss (Linv)')
    # lambda_ds in StarGANv2 maps to lambda_wd in WeatherGAN (weather diversity loss)
    # WeatherGAN uses -lambda_wd * Lwd. StarGANv2 uses -lambda_ds * Lds.
    parser.add_argument('--lambda_wd', type=float, default=-1.0, # WeatherGAN paper: -1.0 (for maximization)
                        help='Weight for weather diversity loss (Lwd)')
    # parser.add_argument('--ds_iter', type=int, default=100000, # Kept from StarGANv2, for decaying lambda_wd if needed.
    #                     help='Number of iterations to optimize diversity sensitive loss') # Not mentioned in WeatherGAN
    parser.add_argument('--w_hpf', type=float, default=0, # WeatherGAN does not mention HPF, setting to 0.
                        help='weight for high-pass filtering')
    # New lambda for WeatherGAN's Lseg
    parser.add_argument('--lambda_seg', type=float, default=1.0, # WeatherGAN paper: 1.0
                        help='Weight for weakly supervised multitask weather-cue objective (Lseg = Lc + Ls)')
    # New lambda for WeatherGAN's Lwrc
    parser.add_argument('--lambda_wrc', type=float, default=1.0, # WeatherGAN paper: 1.0
                        help='Weight for weather-irrelevant content reconstruction loss (Lwrc)')
    
    # Adversarial loss weight (typically 1, but can be tuned)
    parser.add_argument('--lambda_adv', type=float, default=1.0,
                        help='Weight for adversarial loss')


    # Remove StarGANv2 specific lambdas if they are replaced or not used
    # parser.add_argument('--lambda_sty', type=float, default=1, help='Weight for style reconstruction loss') # Replaced by lambda_inv
    # parser.add_argument('--lambda_ds', type=float, default=1, help='Weight for diversity sensitive loss') # Replaced by lambda_wd
    # parser.add_argument('--lambda_wing', type=float, default=0, help='Weight for wing loss (face landmarks)') # Not in WeatherGAN

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=1,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'sample', 'eval', 'align'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='/kaggle/input/five-weather-23k',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='/kaggle/input/five-weather-23k',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--sample_every', type=int, default=99999)
    parser.add_argument('--save_every', type=int, default=2500)
    parser.add_argument('--eval_every', type=int, default=99999)

    # wandb
    parser.add_argument('--wandb_api_token', type=str, default=None)
    parser.add_argument('--wandb_name', type=str, required=True)
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_resume', type=bool, default=False)
    

    parser.add_argument('--max_per_class', type=int, default=2200)
    # parser.add_argument('--lambda_alpha', type=float, default=1.0) # Not in WeatherGAN
    # parser.add_argument('--lambda_beta', type=float, default=1.0) # Potentially superseded by Lcyc and Linv structure

    args = parser.parse_args()
    main(args)
