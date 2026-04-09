# GLU Release Repository

This repository contains a compact public training and evaluation path for GLU:

- stage-0 sparse reconstruction training
- stage-1 latent forecasting training
- unified HDF5 data loading

## Repository Layout

```text
GLU_Reconstruction/
├── Dataset/
│   └── collinear_flow_Re40/
├── configs/
│   ├── collinear_re40_stage0.yaml
│   └── collinear_re40_stage1.yaml
├── outputs/
│   ├── checkpoints/
│   ├── configs/
│   ├── eval/
│   └── loss_history/
├── src/
│   ├── dataloading.py
│   └── models.py
├── train.py
├── evaluate.py
└── requirements.txt
```

## Dataset Format

Each dataset is stored in one HDF5 file with keys:

- `fields`: `[B, N_t, N_x, N_y, N_z, C]`
- `coordinates`: `[N_x, N_y, N_z, N_dim]`
- `time`: `[N_t]`
- optional `conditions`: `[B, N_para]`
- optional `orig_mean`, `orig_std`

For 2D cases, use `N_z = 1`.

## Environment

Install the required Python packages from `requirements.txt` in your runtime environment before training or evaluation.

Optional packages:

- `torchdiffeq`
- `fairscale`

These are only needed for older or non-default code paths and are not required by the shipped configs.

## Training Workflow

`train.py` selects the training stage from the YAML config you pass in.

If you do not provide `--config`, it defaults to:

```bash
python train.py --dataset collinear_re40
```

which resolves to:

```text
configs/collinear_re40_stage0.yaml
```

### Stage 0: Reconstruction

```bash
python train.py --config configs/collinear_re40_stage0.yaml
```

This trains the encoder and reconstructor, including predictive uncertainty and the adaptive importance field.

### Stage 1: Forecasting

```bash
python train.py --config configs/collinear_re40_stage1.yaml
```

This loads the stage-0 checkpoint, freezes the encoder and reconstructor, and trains the temporal forecasting module.

The expected stage-0 checkpoint name is:

```text
outputs/checkpoints/Net_GLU_id0_st0_num0.pth
```

## Main Config Keys

The stage is controlled by the YAML key:

- `Stage: 0` for reconstruction
- `Stage: 1` for forecasting

Other commonly edited keys:

- `data_h5`: dataset path
- `device_ids`: GPU ids
- `batch_size`
- `num_epochs`
- `monitor_every`
- `save_metric`
- `stage0_checkpoint_name` for stage 1

## Outputs

During training, the script writes:

- checkpoints to `outputs/checkpoints/`
- a snapshot of the loaded config to `outputs/configs/`
- loss history to `outputs/loss_history/{net_name}/`

Loss history includes:

- `loss_history.json`
- `overall_loss_history.png`

These are updated every `monitor_every` epochs and include overall loss plus all logged train and held-out sub-terms.

If `save_imp_dists: true` is enabled in the config, the trainer also saves first-case uncertainty and importance visualizations to:

```text
outputs/eval/Imp_{net_name}/
```

with save frequency controlled by:

- `imp_dist_every`
- `imp_dist_split`

## Evaluation

After training, run:

```bash
python evaluate.py --config configs/collinear_re40_stage1.yaml --checkpoint outputs/checkpoints/Net_GLU_id0_st1_num0.pth
```

Evaluation writes:

- `outputs/eval/metrics.json`
- qualitative plot images under `outputs/eval/`

## Notes

- Checkpointing is validation-based by default and the shipped configs use `save_metric: val_mse`.
- The current loader split is train plus held-out. In the code and logs, that held-out split is sometimes referred to as validation.
