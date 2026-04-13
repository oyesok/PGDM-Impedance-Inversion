import os
import torch
from torch.utils.data import DataLoader
from torchvision import utils
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
from scipy.io import savemat


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

def load_model(trainer, model_path, device="cpu"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if "ema" not in checkpoint:
        raise KeyError("Checkpoint missing key: 'ema'")
    trainer.ema_model.load_state_dict(checkpoint["ema"], strict=True)

def split_into_patches(image, patch_size=128, overlap=0):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if len(image.shape) == 2:
        image = image.unsqueeze(0)

    _, H, W = image.shape
    patches = []
    positions = []

    stride = patch_size - overlap
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_start = y
            y_end = min(y + patch_size, H)
            x_start = x
            x_end = min(x + patch_size, W)

            patch = image[:, y_start:y_end, x_start:x_end]

            if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                pad_y = patch_size - patch.shape[1]
                pad_x = patch_size - patch.shape[2]
                patch = torch.nn.functional.pad(patch, (0, pad_x, 0, pad_y), mode='constant', value=0)

            patches.append(patch)
            positions.append((y_start, x_start))

    return patches, positions

def merge_patches(patches, positions, original_shape, patch_size=128, overlap=0):
    if len(original_shape) == 2:
        original_shape = (1,) + original_shape

    merged_image = torch.zeros(original_shape)
    count = torch.zeros(original_shape)

    stride = patch_size - overlap
    for patch, (y_start, x_start) in zip(patches, positions):
        y_end = y_start + patch_size
        x_end = x_start + patch_size

        y_end = min(y_end, original_shape[1])
        x_end = min(x_end, original_shape[2])

        patch_height = y_end - y_start
        patch_width = x_end - x_start
        patch = patch[:, :patch_height, :patch_width]

        print(f"Patch shape: {patch.shape}, Merged region: {merged_image[:, y_start:y_end, x_start:x_end].shape}")

        merged_image[:, y_start:y_end, x_start:x_end] += patch
        count[:, y_start:y_end, x_start:x_end] += 1

    merged_image /= count

    return merged_image

def test(trainer, test_dataloader, patch_size=128):
    trainer.model.eval()
    trainer.ema_model.eval()

    for i, batch in enumerate(test_dataloader):
        seismic = batch['seismic'].squeeze().cpu().numpy()
        initial_model = batch['initial_model'].squeeze().cpu().numpy()
        labels = batch['label'].cuda()

        original_size = seismic.shape
        padded_size = (
            int(np.ceil(original_size[0] / patch_size) * patch_size),
            int(np.ceil(original_size[1] / patch_size) * patch_size)
        )


        padded_seismic = np.pad(seismic,
                                     [(0, padded_size[0] - original_size[0]), (0, padded_size[1] - original_size[1])],
                                     mode='reflect')
        padded_initial = np.pad(initial_model,
                                      [(0, padded_size[0] - original_size[0]), (0, padded_size[1] - original_size[1])],
                                      mode='reflect')

        seismic_tensor = torch.tensor(padded_seismic, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        initial_tensor = torch.tensor(padded_initial, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

        x1 = torch.cat([initial_tensor, seismic_tensor], dim=1)
        with torch.no_grad():
            xt, _, _, _ = trainer.ema_model.module.sample(
                condition=x1,
                T_total=trainer.ema_model.module.num_timesteps,
                device=trainer.ema_model.module.device,
                t=None
            )

        xt = xt.squeeze().cpu().numpy()[:original_size[0], :original_size[1]]

        max_value = 12088
        min_value = 1136

        predicted_impedance = (xt + 1) * (max_value - min_value) / 2 + min_value

        save_path = f'results/predicted_impedance.mat'
        savemat(save_path, {'predicted_impedance': predicted_impedance})
        print(f"predicted_impedance saved to {save_path}")

        save_with_colormap(predicted_impedance, f'results/predicted_impedance.png', nrow=1)
        save_with_colormap(labels, f'results/true_impedance.png', nrow=1)

def evaluate(predicted, target):
    mse = F.mse_loss(predicted, target).item()
    mae = F.l1_loss(predicted, target).item()
    print(f'MSE: {mse}, MAE: {mae}')


if __name__ == '__main__':
    from Trainer import Trainer
    from dataset import WaveImpedanceDataset
    from Unet import Unet
    from GaussianDiffusion import GaussianDiffusion

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        out_dim=1,
        with_time_emb=True,
        residual=False
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        channels=2,
        timesteps=1000,
        eta=0.0,
        loss_type='l1_ms_ssim',
        train_routine='Final',
        sampling_routine='default'
    ).cuda()

    diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

    trainer = Trainer(
        diffusion,
        seismic_data_dir='./testdata',
        impedance_data_dir='./testdata',
        initial_model_dir='./testdata',
        image_size=128,
        train_batch_size=16,
        train_lr=1e-4,
        train_num_steps=1000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        fp16=False,
        results_folder='results',
        load_path= None,
        dataset='test',
        save_and_sample_every=100
    )

    load_model(trainer, './checkpoints/model_test.pt')

    test_dataset = WaveImpedanceDataset(
        seismic_data_dir='./testdata',
        impedance_data_dir='./testdata',
        initial_model_dir='./testdata',
        root_dir='./testdata',
        mode='test'
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test(trainer, test_dataloader, patch_size=128)

    # trainer.paper_showing_diffusion_images_cover_page()
