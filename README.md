# PGDM-Impedance-Inversion

This repository contains the code for seismic impedance inversion.

## Files

- `dataset.py` — Data loading
- `GaussianDiffusion.py` — Model definition
- `Unet.py` — Network architecture
- `Trainer.py` — Training utilities
- `main.py` — Training script
- `test.py` — Testing script

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- numpy
- scipy
- matplotlib
- einops
- pytorch_msssim
- scikit-learn
- comet_ml

## Training

```bash
python main.py
```

## Testing / Inference

A pretrained checkpoint for testing is available on Hugging Face::
https://huggingface.co/yes11-ok/PGDM-Impedance-Inversion

### Step 1. Download pretrained checkpoint

Download `model_test.pt` from Hugging Face.

### Step 2. Place the checkpoint file

Put the downloaded checkpoint in:

```bash
./checkpoints/model_test.pt
```
### Step 3. Run Testing

```bash
python test.py
```

## Outputs

Training results and checkpoints are saved in:
```bash
./results_wave_impedance
```
Testing results are saved in:
```bash
./results
```
