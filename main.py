from GaussianDiffusion import GaussianDiffusion
from Unet import Unet
from Trainer import Trainer
import torch
import os
import errno
import shutil
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from multiprocessing import freeze_support

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', default=1000, type=int)
    parser.add_argument('--train_steps', default=15000, type=int)
    parser.add_argument('--save_folder', default='./results_wave_impedance', type=str)
    parser.add_argument('--seismic_data_path', default='./traindata/seismic data', type=str)
    parser.add_argument('--impedance_data_path', default='./traindata/impedance', type=str)
    parser.add_argument('--initial_model_path', default='./traindata/initial model', type=str)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--train_routine', default='Final', type=str)
    parser.add_argument('--sampling_routine', default='default', type=str)
    parser.add_argument('--remove_time_embed', action="store_true")
    parser.add_argument('--residual', action="store_true")
    parser.add_argument('--loss_type', default='l1_ms_ssim', type=str)

    args = parser.parse_args()
    print(args)

    device_id = 0
    torch.cuda.set_device(device_id)

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        out_dim= 1,
        with_time_emb=not(args.remove_time_embed),
        residual=args.residual
    ).cuda()

    diffusion = GaussianDiffusion(
        denoise_fn=model,
        loss_type='l1_ms_ssim',
        channels=2,
        image_size=128,
        eta= 0.0,
        timesteps=args.time_steps
    ).cuda()

    diffusion = torch.nn.DataParallel(diffusion, device_ids=[0], output_device=0)

    trainer = Trainer(
        diffusion,
        args.seismic_data_path,
        args.impedance_data_path,
        initial_model_dir=args.initial_model_path,
        image_size=128,
        train_batch_size=8,
        train_lr=1e-4,
        train_num_steps=args.train_steps,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        fp16=False,
        results_folder=args.save_folder,
        load_path=args.load_path,
        dataset='train',
        save_and_sample_every=100
    )

    trainer.train()

if __name__ == '__main__':
    freeze_support()
    main()