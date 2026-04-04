# trainer class
from comet_ml import Experiment

import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F

from functools import partial
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils

from sklearn.mixture import GaussianMixture
import os
import errno
from torch.utils.data import DataLoader
from Unet import EMA
from dataset import WaveImpedanceDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

from collections import OrderedDict
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k.replace('.module', '')  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict

def adjust_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k.replace('denoise_fn.module', 'module.denoise_fn')  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def loss_backwards(fp16, loss, optimizer, scaler=None, **kwargs):
    if fp16:
        if scaler is not None:
            scaler.scale(loss).backward(**kwargs)
        else:
            raise RuntimeError("Scaler is required for FP16 training.")
    else:
        loss.backward(**kwargs)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def save_with_colormap(images, path, colormap='jet', nrow=4):
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    if images.ndim == 4:
        batch_size, num_channels, height, width = images.shape

        if num_channels > 1:
            for c in range(num_channels):
                channel_images = images[:, c, :, :]

                num_images = batch_size
                ncol = min(nrow, num_images)
                nrow_images = (num_images + ncol - 1) // ncol

                fig, axes = plt.subplots(nrow_images, ncol, figsize=(ncol * 3, nrow_images * 3))
                if nrow_images == 1 and ncol == 1:
                    axes = np.array([[axes]])
                elif nrow_images == 1 or ncol == 1:
                    axes = axes.reshape(nrow_images, ncol)

                for i, img in enumerate(channel_images):
                    row = i // ncol
                    col = i % ncol
                    ax = axes[row, col]
                    ax.imshow(img, cmap=colormap)
                    ax.axis('off')

                for i in range(num_images, nrow_images * ncol):
                    row = i // ncol
                    col = i % ncol
                    ax = axes[row, col]
                    ax.axis('off')

                plt.tight_layout()
                channel_path = f"{path}_channel_{c}.png"
                plt.savefig(channel_path, bbox_inches='tight', pad_inches=0)
                plt.close()
        else:
            images = images.squeeze(1)

            num_images = batch_size
            ncol = min(nrow, num_images)
            nrow_images = (num_images + ncol - 1) // ncol

            fig, axes = plt.subplots(nrow_images, ncol, figsize=(ncol * 3, nrow_images * 3))
            if nrow_images == 1 and ncol == 1:
                axes = np.array([[axes]])
            elif nrow_images == 1 or ncol == 1:
                axes = axes.reshape(nrow_images, ncol)

            for i, img in enumerate(images):
                row = i // ncol
                col = i % ncol
                ax = axes[row, col]
                ax.imshow(img, cmap=colormap)
                ax.axis('off')

            for i in range(num_images, nrow_images * ncol):
                row = i // ncol
                col = i % ncol
                ax = axes[row, col]
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close()
    else:
        if images.ndim == 3 and images.shape[0] > 1:
            for c in range(images.shape[0]):
                channel_img = images[c]
                channel_path = f"{path}_channel_{c}.png"
                plt.imshow(channel_img, cmap=colormap)
                plt.axis('off')
                plt.savefig(channel_path, bbox_inches='tight', pad_inches=0)
                plt.close()
        else:
            if images.ndim == 3:
                images = images.squeeze(0)
            plt.imshow(images, cmap=colormap)
            plt.axis('off')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close()

def save_single_map(tensor_hw, out_path, cmap='jet', vmin=None, vmax=None):
    if isinstance(tensor_hw, torch.Tensor):
        arr = tensor_hw.detach().squeeze().float().cpu().numpy()
    else:
        arr = np.squeeze(tensor_hw)

    if arr.ndim != 2:
        raise ValueError(f"save_single_map expects 2D array, got shape {arr.shape}")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Patch is not square: {arr.shape}. Expected 128x128.")

    plt.figure(figsize=(3, 3))
    im = plt.imshow(arr, cmap=cmap, origin='upper',
                    aspect='equal',
                    vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig(out_path, dpi=220, bbox_inches='tight', pad_inches=0)
    plt.close()

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        seismic_data_dir,
        impedance_data_dir,
        initial_model_dir,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 16,
        train_lr = 1e-4,
        train_num_steps = 15000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 100,
        results_folder = './results',
        load_path = None,
        dataset = None,
        shuffle=True
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        if dataset == 'train':
            train_dataset = WaveImpedanceDataset(
                seismic_data_dir=seismic_data_dir,
                impedance_data_dir=impedance_data_dir,
                initial_model_dir=initial_model_dir,
                root_dir='./traindata',
                mode='train',
                split_ratio=0.8,
                seed=42
            )

            val_dataset = WaveImpedanceDataset(
                seismic_data_dir=seismic_data_dir,
                impedance_data_dir=impedance_data_dir,
                initial_model_dir=initial_model_dir,
                root_dir='./traindata',
                mode='val',
                split_ratio=0.8,
                seed=42
            )

            # DataLoader
            self.dl = cycle(DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                pin_memory=True,
                num_workers=0,
                drop_last=True
            ))

            self.val_dl = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0,
                drop_last=False
            )

        elif dataset == 'test':
            # Dataset
            self.ds = WaveImpedanceDataset(
                seismic_data_dir=seismic_data_dir,
                impedance_data_dir=impedance_data_dir,
                initial_model_dir=initial_model_dir,
                root_dir='./testdata',
                mode='test',
                seed=42
            )

            self.dl = cycle(DataLoader(
                self.ds,
                batch_size=self.batch_size,
                shuffle=shuffle,
                pin_memory=True,
                num_workers=0,
                drop_last=True
            ))

            self.val_dl = None

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.fp16 = fp16

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)

        # --- for local logging (loss curves) ---
        self.metrics_csv = str(self.results_folder / "metrics.csv")
        with open(self.metrics_csv, "w", encoding="utf-8") as f:
            f.write("step,train_loss,val_loss\n")
        self._loss_hist = {"step": [], "train": [], "val": []}

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, itrs=None):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'opt': self.opt.state_dict()
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / f'model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model_{itrs}.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def add_title(self, path, title):

        import cv2
        import numpy as np

        img1 = cv2.imread(path)

        # --- Here I am creating the border---
        black = [0, 0, 0]  # ---Color of the border---
        constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(vcat, str(title), (violet.shape[1] // 2, height-2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat)

    def validate(self):
        self.ema_model.eval()
        total_loss = 0
        val_loader = self.val_dl

        with torch.no_grad():
            for batch in val_loader:
                seismic = batch['seismic'].cuda()
                initial_model = batch['initial_model'].cuda()
                labels = batch['label'].cuda()

                x1 = torch.cat([initial_model, seismic], dim=1)

                loss = torch.mean(self.model(labels, x1))
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def visualize_training_patch(self, batch, vis_idx=0,
                                 steps=(0, 30, 63, 250, 500, 800, 999),
                                 outdir=None):

        import os
        os.makedirs(outdir, exist_ok=True)

        seismic = batch['seismic'].cuda()
        initial_model = batch['initial_model'].cuda()
        labels = batch['label'].cuda()
        B = initial_model.shape[0]

        with torch.no_grad():
            (S_syn_list, mix_list, noisy_list,
             Backward, final_img, S_syn_pred_list, r_input_list) = self.ema_model.module.forward_and_backward(
                batch_size=B, img1=initial_model, img2=seismic, img3=labels)

        imp_a = initial_model[vis_idx]
        imp_b = labels[vis_idx]
        vmin_imp = torch.minimum(imp_a.min(), imp_b.min()).item()
        vmax_imp = torch.maximum(imp_a.max(), imp_b.max()).item()

        seis0 = seismic[vis_idx]
        vmin_seis = seis0.min()
        vmax_seis = seis0.max()
        for t in steps:
            vmin_seis = torch.minimum(vmin_seis, S_syn_list[t][vis_idx].min())
            vmax_seis = torch.maximum(vmax_seis, S_syn_list[t][vis_idx].max())
        vmin_seis = vmin_seis.item()
        vmax_seis = vmax_seis.item()

        save_single_map(initial_model[vis_idx], f"{outdir}/initial_model.png", cmap='jet', vmin=vmin_imp, vmax=vmax_imp)
        save_single_map(labels[vis_idx], f"{outdir}/gt.png", cmap='jet', vmin=vmin_imp, vmax=vmax_imp)
        arr = seismic[vis_idx].detach().squeeze().cpu().numpy()
        max_abs = float(np.nanmax(np.abs(arr)))
        save_single_map(seismic[vis_idx], f"{outdir}/seismic.png", cmap='seismic', vmin=-max_abs, vmax=max_abs)

        for t in steps:
            # r_input_list[t][vis_idx]: (1, H-1, W)
            r_to_show = F.pad(r_input_list[t][vis_idx], (0, 0, 0, 1))  # → (1, H, W)
            r_to_show1 = r_to_show[vis_idx].detach().squeeze().cpu().numpy()
            max_abs1 = float(np.nanmax(np.abs(r_to_show1)))
            save_single_map(r_to_show[0], f"{outdir}/forward_r{t}.png", cmap='seismic', vmin=-max_abs1, vmax=max_abs1)
            save_single_map(S_syn_list[t][vis_idx], f"{outdir}/forward_Ssyn_t{t}.png", cmap='seismic', vmin=-max_abs, vmax=max_abs)
            save_single_map(noisy_list[t][vis_idx], f"{outdir}/forward_xt_t{t}.png", cmap='jet')

        T = len(Backward)  # = num_timesteps
        for t in reversed(steps):
            if t == T - 1:
                continue
            rev_idx = (T - 1) - t
            save_single_map(Backward[rev_idx][vis_idx], f"{outdir}/backward_recon_t{t}.png", cmap='jet')

        for t in reversed(steps):
            if t == T - 1:
                continue
            rev_idx = (T - 1) - t
            save_single_map(S_syn_pred_list[rev_idx][vis_idx],
                            f"{outdir}/backward_Ssynpred_t{t}.png", cmap='seismic', vmin=-max_abs, vmax=max_abs)

    def visualize_sample_by_index(self, dataset=None, index=0,
                                  steps=(0, 30, 63, 125, 249, 499),
                                  outdir="vis_fixed"):

        import os, torch, warnings
        os.makedirs(outdir, exist_ok=True)

        device = getattr(getattr(self, 'ema_model', None), 'module', None)
        device = getattr(device, 'device', None) or next(self.ema_model.parameters()).device

        if dataset is None:
            ds = None

            gen = getattr(self, 'dl', None)
            try:
                if gen is not None and hasattr(gen, 'gi_frame') and gen.gi_frame is not None:
                    f_locals = gen.gi_frame.f_locals
                    if 'dl' in f_locals:
                        dl_underlying = f_locals['dl']
                        ds = getattr(dl_underlying, 'dataset', None)
            except Exception:
                ds = None

            if ds is None and hasattr(self, 'val_dl') and self.val_dl is not None:
                ds = getattr(self.val_dl, 'dataset', None)
            if ds is None and hasattr(self, 'ds'):
                ds = getattr(self, 'ds', None)

            dataset = ds

        if dataset is None:
            warnings.warn(
                "[visualize_sample_by_index] Failed to access the dataset. Falling back to visualizing the first sample in the current batch (index will be ignored)."
            )
            batch_any = next(self.dl)
            batch = {
                'seismic': batch_any['seismic'][0:1].to(device),
                'initial_model': batch_any['initial_model'][0:1].to(device),
                'label': batch_any['label'][0:1].to(device),
            }

            T = getattr(self.ema_model.module, 'num_timesteps', None)
            if T is not None:
                steps = tuple(sorted({s for s in steps if 0 <= s < T}))
                if len(steps) == 0:
                    steps = (0, T // 2, T - 1)
            return self.visualize_training_patch(batch, vis_idx=0, steps=steps, outdir=outdir)

        if index < 0 or index >= len(dataset):
            raise IndexError(f"index {index} out of range (len={len(dataset)})")

        sample = dataset[index]
        batch = {
            'seismic': sample['seismic'].unsqueeze(0).to(device),
            'initial_model': sample['initial_model'].unsqueeze(0).to(device),
            'label': sample['label'].unsqueeze(0).to(device),
        }

        T = getattr(self.ema_model.module, 'num_timesteps', None)
        if T is not None:
            steps = tuple(sorted({s for s in steps if 0 <= s < T}))
            if len(steps) == 0:
                steps = (0, T // 2, T - 1)

        self.visualize_training_patch(batch, vis_idx=0, steps=steps, outdir=outdir)

    def train(self):
        experiment = Experiment(api_key="57ArytWuo2X4cdDmgU1jxin77",
                                project_name="Cold_Diffusion_Cycle")

        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0
        best_val_loss = float('inf')
        patience = 50
        no_improvement_count = 0

        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dl)
                seismic = batch['seismic'].cuda()
                initial_model = batch['initial_model'].cuda()
                labels = batch['label'].cuda()

                condition = torch.cat([initial_model, seismic], dim=1)  # [B, 2, H, W]

                loss = torch.mean(self.model(labels, condition))

                if self.step % 100 == 0:
                    print(f'{self.step}: {loss.item()}')
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)

            acc_loss = acc_loss + (u_loss/self.gradient_accumulate_every)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                print(f"Saving model and samples at step {self.step}")
                experiment.log_current_epoch(self.step)
                milestone = self.step // self.save_and_sample_every
                batches = self.batch_size

                batch = next(self.dl)
                seismic = batch['seismic'].cuda()
                initial_model = batch['initial_model'].cuda()

                x1 = torch.cat([initial_model, seismic], dim=1)


                fixed_idx = 678
                self.visualize_sample_by_index(index=fixed_idx,
                                               steps=(0, 30, 63, 250, 500, 800, 999),
                                               outdir=str(
                                                   self.results_folder / f"train_vis_fixed_{fixed_idx}_step_{self.step}"))

                with torch.no_grad():
                    self.ema_model.eval()

                    xt, direct_recons, S_syn_pred, seismic = self.ema_model.module.sample(
                        condition=x1,
                        T_total=self.ema_model.module.num_timesteps,
                        device=self.ema_model.module.device,
                        t=None
                    )

                all_images = (xt + 1) * 0.5
                print(f"target_pred shape: {all_images.shape}")
                save_with_colormap(all_images, str(self.results_folder / f'sample-recon-{milestone}.png'))

                labels = (labels + 1) * 0.5  #
                print(f"labels shape: {labels.shape}")
                save_with_colormap(labels, str(self.results_folder / f'sample-labels-{milestone}.png'))

                K_t = (S_syn_pred + 1) * 0.5  #
                print(f"K_t shape: {K_t.shape}")
                save_with_colormap(K_t, str(self.results_folder / f'S_syn_pred-{milestone}.png'))

                seismic = (seismic + 1) * 0.5  #
                print(f"seismic shape: {seismic.shape}")
                save_with_colormap(seismic, str(self.results_folder / f'seismic-{milestone}.png'))

                del all_images
                torch.cuda.empty_cache()

                acc_loss = acc_loss/(self.save_and_sample_every+1)
                experiment.log_metric("Training Loss", acc_loss, step=self.step)
                print(f'Mean of last {self.step}: {acc_loss}')

                if self.val_dl is not None:
                    val_loss = self.validate()
                    val_loss_to_log = val_loss
                    experiment.log_metric("Validation Loss", val_loss, step=self.step)
                    print(f'Validation Loss: {val_loss}')

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save('best_model.pth')
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= patience:
                        print(f"Early stopping at step {self.step} due to no improvement in validation loss.")
                        break

                self._loss_hist["step"].append(self.step)
                self._loss_hist["train"].append(acc_loss)
                self._loss_hist["val"].append(val_loss_to_log)

                with open(self.metrics_csv, "a", encoding="utf-8") as f:
                    f.write(
                        f"{self.step},{acc_loss},{'' if (val_loss_to_log != val_loss_to_log) else val_loss_to_log}\n")

                acc_loss = 0

                self.save()
                if self.step % (self.save_and_sample_every * 100) == 0:
                    self.save(self.step)

            self.step += 1

        print('training completed')

        steps = self._loss_hist["step"]
        train_losses = self._loss_hist["train"]
        val_losses = self._loss_hist["val"]

        plt.figure(figsize=(7, 4.2))
        plt.plot(steps, train_losses, label="Train Loss", linewidth=2)

        if any([v == v for v in val_losses]):
            vs = [v for v in val_losses]
            plt.plot(steps, vs, label="Val Loss", linewidth=2)

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training / Validation Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_png = str(self.results_folder / "loss_curves.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[Saved] {out_png}")


    def sample_and_save_for_fid(self, noise=0):

        out_folder = f'{self.results_folder}_out'
        create_folder(out_folder)


        cnt = 0
        bs = self.batch_size
        for j in range(int(6400/self.batch_size)):

            batch = next(self.dl)
            seismic = batch['seismic'].cuda()
            initial_model = batch['initial_model'].cuda()
            x1 = torch.cat([initial_model, seismic], dim=1)

            xt, direct_recons, all_images = self.ema_model.module.gen_sample(batch_size=bs, img=x1,
                                                                             noise_level=noise)

            for i in range(all_images.shape[0]):
                utils.save_image((all_images[i] + 1) * 0.5,
                                 str(f'{out_folder}/' + f'sample-x0-{cnt}.png'))


                cnt += 1

    def paper_showing_diffusion_images_cover_page(self):
        import cv2
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from torch.utils.data import DataLoader

        cnt = 0
        to_show = [0, 30, 63, 250, 500, 800, 999]

        dataloader = DataLoader(
            self.ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )

        for i, batch in enumerate(dataloader):
            if i >= 5:
                break

            init_m = batch['initial_model'].cuda()
            seis = batch['seismic'].cuda()
            tgt  = batch['label'].cuda()
            B = init_m.shape[0]

            _, _, noisy_list, Backward, final_all, _ ,_= self.ema_model.module.forward_and_backward(
                batch_size=B, img1=init_m, img2=seis, img3=tgt)

            for k in range(B):
                print(f"Saving sample {cnt}, in-batch index {k}")
                images = []

                # Helper: convert tensor [1,H,W] to normalized numpy array
                def to_np(x):
                    return x.squeeze().cpu().numpy().astype(np.float32)

                for j in to_show:
                    arr = to_np(noisy_list[j][k])
                    fig, ax = plt.subplots(frameon=False)
                    ax.imshow(arr, cmap='jet_r', aspect='auto')
                    ax.axis('off')
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    fig.canvas.draw()
                    imgN = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    imgN = imgN.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.close(fig)
                    images.append(imgN)

                T = len(Backward)  # == self.num_timesteps
                for j in reversed(to_show):
                    if j == T - 1:
                        continue

                    rev_idx = (T - 1) - j
                    arr = to_np(Backward[rev_idx][k])

                    fig, ax = plt.subplots(frameon=False)
                    ax.imshow(arr, cmap='jet_r', aspect='auto')
                    ax.axis('off')
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    fig.canvas.draw()
                    imgB = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    imgB = imgB.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.close(fig)
                    images.append(imgB)

                row = cv2.hconcat(images)
                cv2.imwrite(str(self.results_folder / f'all_{cnt}.png'), row)

                cnt += 1

            del noisy_list, Backward, final_all
            print(f"Completed sample {cnt}")


