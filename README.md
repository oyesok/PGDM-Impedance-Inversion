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

## Run Training

```bash
python main.py
```

## Run Testing

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
