from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import random
import shutil
import sys
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# this script lives at repo root and imports from ./src
repo_dir = pathlib.Path(__file__).resolve().parent
src_dir = repo_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from src.dataloading import make_loaders, make_loaders_DSUS  
from src.models import (  
    GLU_Bay_DD,
    DomainAdaptiveEncoder,
    TemporalDecoderSoftmax,
    TemporalDecoderHierarchical,
    SoftDomainAdaptiveReconstructor,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='collinear_flow_Re40', type=str)
    p.add_argument('--config', type=pathlib.Path, help='Path to YAML config.')
    args = p.parse_args()
    if args.config is None:
        args.config = pathlib.Path(f'configs/{args.dataset}_stage0.yaml')
    return args


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def load_cfg(yaml_path: pathlib.Path) -> Dict:
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    time_tag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    snap_dir = repo_dir / 'outputs' / 'configs'
    snap_dir.mkdir(parents=True, exist_ok=True)
    # Save the exact config used for this run so evaluation can be reproduced later.
    shutil.copyfile(
        yaml_path,
        snap_dir / f"{yaml_path.stem}_id{cfg['case_index']}_st{cfg['Stage']}_num{cfg['Repeat_id']}_{time_tag}.yaml",
    )
    # Fill in release-friendly defaults so minimal YAMLs still run end to end.
    cfg.setdefault('seed', 42)
    cfg.setdefault('warmup_epochs', 20)
    cfg.setdefault('weight_decay', 0.0)
    cfg.setdefault('grad_clip', 1.0)
    cfg.setdefault('teacher_force_start', 0.75)
    cfg.setdefault('teacher_force_decay', 1000.0)
    cfg.setdefault('teacher_force_min', 0.05)
    cfg.setdefault('nll_anneal_epochs', 100)
    cfg.setdefault('nll_weight', 1.0)
    cfg.setdefault('monitor_every', 10)
    cfg.setdefault('patience_epochs', 100)
    cfg.setdefault('Loss_cls_Weight', 0.10)
    cfg.setdefault('Supervise_Sensors', False)
    cfg.setdefault('BATCH_DOWNSAMPLE', False)
    cfg.setdefault('DOWNSAMPLE_LOGIC', 'random')
    cfg.setdefault('batch_downsample_min', 0.50)
    cfg.setdefault('batch_downsample_max', 1.00)
    cfg.setdefault('temporal_uncert_weight', 0.0)
    cfg.setdefault('save_metric', 'val_mse')
    cfg.setdefault('save_monitor_checkpoints', True)
    cfg.setdefault('save_imp_dists', False)
    cfg.setdefault('imp_dist_every', 1000)
    cfg.setdefault('imp_dist_split', 'val')
    cfg.setdefault('imp_dist_max_batches', 1)
    cfg.setdefault('bayesian_phi', {})
    b = cfg['bayesian_phi']
    b.setdefault('phi_mlp_hidden_dim', 128)
    b.setdefault('prior_alpha', 2.0)
    b.setdefault('prior_beta', 5.0)
    b.setdefault('mc_samples_elbo', 5)
    b.setdefault('vi_entropy_weight', 0.02)
    b.setdefault('var_weight', 0.20)
    b.setdefault('lambda_kl', 0.01)
    b.setdefault('lambda_elbo', 5e-5)
    b.setdefault('anneal_epochs', 100)
    b.setdefault('update_in_stage1', True)
    return cfg


def teacher_force_prob(epoch: int, cfg: Dict) -> float:
    return max(cfg['teacher_force_min'], cfg['teacher_force_start'] - epoch / cfg['teacher_force_decay'])


def make_data_loaders(cfg: Dict):
    use_dsus = cfg.get('USE_DSUS', False)
    if use_dsus:
        return make_loaders_DSUS(
            cfg['data_h5'],
            num_time_sample=cfg['num_time_sample'],
            num_space_sample=cfg['num_space_sample'],
            Num_x=cfg['Num_x'],
            Num_y=cfg['Num_y'],
            global_downsample_ratio=cfg['global_downsample_ratio'],
            multi_factor=cfg['multi_factor'],
            train_ratio=cfg['train_ratio'],
            batch_size=cfg['batch_size'],
            workers=cfg['num_workers'],
            channel=cfg['channel'],
            process_mode=cfg['process_mode'],
            num_samples=cfg['num_samples'],
            Full_Field_DownS=cfg['Full_Field_DownS'],
            global_restriction=cfg['global_restriction'],
            sample_restriction=cfg['sample_restriction'],
            sample_params=cfg['Sample_Parameters'],
        )
    return make_loaders(
        cfg['data_h5'],
        num_time_sample=cfg['num_time_sample'],
        num_space_sample=cfg['num_space_sample'],
        multi_factor=cfg['multi_factor'],
        train_ratio=cfg['train_ratio'],
        batch_size=cfg['batch_size'],
        workers=cfg['num_workers'],
        channel=cfg['channel'],
        process_mode=cfg['process_mode'],
        num_samples=cfg['num_samples'],
        Full_Field_DownS=cfg['Full_Field_DownS'],
        global_restriction=cfg['global_restriction'],
        sample_restriction=cfg['sample_restriction'],
        sample_params=cfg['Sample_Parameters'],
    )


def build_model(cfg: Dict, n_c: int) -> Tuple[GLU_Bay_DD, str]:
    net_name = f"GLU_id{cfg['case_index']}_st{cfg['Stage']}_num{cfg['Repeat_id']}"
    encoder = DomainAdaptiveEncoder(
        All_dim=cfg['F_dim'],
        num_heads=cfg['num_heads'],
        latent_layers=cfg['num_layers'],
        N_channels=n_c,
        num_freqs=cfg['num_freqs'],
        latent_tokens=cfg['latent_tokens'],
        pooling=cfg['pooling'],
        retain_cls=cfg.get('retain_cls', False),
        retain_lat=cfg.get('retain_lat', False),
        channel_to_encode=cfg.get('channel_to_encode', None),
    )
    if cfg['decoder_type'] == 'CausalTrans':
        temporal = TemporalDecoderSoftmax(
            d_model=cfg['F_dim'],
            n_layers=cfg['num_layers_propagator'],
            n_heads=cfg['num_heads'],
            dt=cfg['delta_t'],
        )
    elif cfg['decoder_type'] == 'UD_Trans':
        temporal = TemporalDecoderHierarchical(
            d_model=cfg['F_dim'],
            n_layers=cfg['num_layers_propagator'],
            n_heads=cfg['num_heads'],
            dt=cfg['delta_t'],
        )
    else:
        raise ValueError(f"Unknown decoder_type: {cfg['decoder_type']}")

    field_dec = SoftDomainAdaptiveReconstructor(
        d_model=cfg['F_dim'],
        num_heads=cfg['num_heads'],
        N_channels=n_c,
        latent_tokens=cfg['latent_tokens'],
        importance_scale=cfg['importance_scale'],
        bandwidth_init=cfg['bandwidth_init'],
        top_k=cfg['top_k'],
        per_sensor_sigma=cfg['per_sensor_sigma'],
        CalRecVar=cfg.get('CalRecVar', False),
        retain_cls=cfg.get('retain_cls', False),
        retain_lat=cfg.get('retain_lat', False),
        USE_FINAL_MLP=cfg.get('USE_FINAL_MLP', False),
    )
    model = GLU_Bay_DD(
        cfg,
        encoder,
        temporal,
        field_dec,
        delta_t=cfg['delta_t'],
        N_window=cfg['N_window'],
        stage=cfg['Stage'],
        use_adaptive_selection=cfg.get('Use_Adaptive_Selection', False),
        CalRecVar=cfg.get('CalRecVar', False),
        retain_cls=cfg.get('retain_cls', False),
        retain_lat=cfg.get('retain_lat', False),
        Use_imp_in_dyn=cfg.get('Use_imp_in_dyn', False),
    )
    return model, net_name


def reinit_temporal_decoder(model: GLU_Bay_DD) -> None:
    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            if getattr(m, 'weight', None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
    model.temporaldecoder.apply(_init)


def maybe_load_stage0_weights(model: GLU_Bay_DD, cfg: Dict) -> None:
    if cfg['Stage'] < 1 and not cfg.get('Reload_Trained', False):
        return
    load_name = cfg.get('stage0_checkpoint_name', f"GLU_id{cfg['case_index']}_st0_num0")
    ckpt_path = repo_dir / cfg['save_net_dir'] / f'Net_{load_name}.pth'
    if not ckpt_path.exists():
        print(f'Warning: no stage-0 checkpoint at {ckpt_path}')
        return
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    # Stage 1 reuses the spatial encoder/reconstructor but starts temporal forecasting fresh.
    filtered = {k: v for k, v in state_dict.items() if not (k.startswith('decoder_lat.') or k.startswith('temporaldecoder.'))}
    model.load_state_dict(filtered, strict=False)
    reinit_temporal_decoder(model)
    print(f'Loaded stage-0 weights from {ckpt_path}')


def freeze_for_stage(model: GLU_Bay_DD, cfg: Dict) -> None:
    stage = cfg['Stage']
    if stage == 0:
        # Stage 0 learns spatial reconstruction only.
        for p in model.temporaldecoder.parameters():
            p.requires_grad = False
        if cfg.get('Use_Adaptive_Selection', False) and hasattr(model, 'phi_mlp_2'):
            for p in model.phi_mlp_2.parameters():
                p.requires_grad = False
    elif stage >= 1:
        # Stage 1 keeps the learned spatial path fixed and optimizes the temporal module.
        for p in model.fieldencoder.parameters():
            p.requires_grad = False
        for p in model.decoder.parameters():
            p.requires_grad = False
        if cfg.get('Use_Adaptive_Selection', False) and hasattr(model, 'phi_mlp_1'):
            for p in model.phi_mlp_1.parameters():
                p.requires_grad = False


def get_channel_weights(cfg: Dict, n_c: int, device: torch.device) -> torch.Tensor:
    weights = cfg.get('channel_weights', [1.0] * n_c)
    if not isinstance(weights, list) or len(weights) != n_c:
        weights = [1.0] * n_c
    return torch.tensor([float(w) for w in weights], dtype=torch.float32, device=device)


def weighted_channel_mse(pred: torch.Tensor, target: torch.Tensor, channel_weights: torch.Tensor):
    losses = [F.mse_loss(pred[..., c], target[..., c], reduction='mean') for c in range(pred.shape[-1])]
    total = sum(channel_weights[c] * losses[c] for c in range(pred.shape[-1]))
    return total, losses


def sensor_supervision_loss(g_u_mean_sens: Optional[torch.Tensor], g_d: torch.Tensor, cfg: Dict, channel_weights: torch.Tensor) -> torch.Tensor:
    if cfg['Stage'] != 0 or (not cfg.get('Supervise_Sensors', False)) or g_u_mean_sens is None:
        return torch.tensor(0.0, device=g_d.device)
    target = g_d[..., 2:2 + g_u_mean_sens.shape[-1]]
    loss, _ = weighted_channel_mse(g_u_mean_sens, target, channel_weights)
    return loss


def heteroscedastic_nll(out: torch.Tensor, out_logvar: Optional[torch.Tensor], target: torch.Tensor, sensor_logvar: Optional[torch.Tensor], cfg: Dict, nll_weight: float) -> torch.Tensor:
    if (not cfg.get('CalRecVar', False)) or cfg['Stage'] != 0 or out_logvar is None:
        return torch.tensor(0.0, device=target.device)
    var = torch.exp(out_logvar) + 1e-6
    nll_main = 0.5 * (((out - target) ** 2) / var + out_logvar).mean()
    nll_sensors = torch.tensor(0.0, device=target.device)
    if cfg.get('Supervise_Sensors', False) and sensor_logvar is not None:
        nll_sensors = (torch.exp(sensor_logvar) + 1e-6).mean()
    return nll_weight * (nll_main + nll_sensors)


def trajectory_loss(obs: torch.Tensor, traj: torch.Tensor, cfg: Dict) -> torch.Tensor:
    if cfg['Stage'] == 0 or cfg['N_window'] >= obs.shape[1]:
        return torch.tensor(0.0, device=obs.device)
    if cfg.get('retain_cls', False):
        loss_cls = F.mse_loss(traj[:, cfg['N_window']:, :1, :], obs[:, cfg['N_window']:, :1, :])
        loss_follow = F.mse_loss(traj[:, cfg['N_window']:, 1:, :], obs[:, cfg['N_window']:, 1:, :])
        return cfg.get('Loss_traj_cls_Weight', 1.0) * loss_cls + loss_follow
    return F.mse_loss(traj[:, cfg['N_window']:, :], obs[:, cfg['N_window']:, :])


def normalize_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + eps)


def compute_elbo(base_model: GLU_Bay_DD, uncert: torch.Tensor, coords: torch.Tensor, cfg: Dict) -> torch.Tensor:
    bcfg = cfg['bayesian_phi']
    log_ab = base_model.phi_mlp_1(coords)
    alpha = torch.exp(log_ab[:, 0]) + 1e-3
    beta = torch.exp(log_ab[:, 1]) + 1e-3
    phi_dist = torch.distributions.Beta(alpha, beta)
    mean_phi = torch.clamp(alpha / (alpha + beta), min=1e-3, max=1 - 1e-3)

    expected_reward = 0.0
    for _ in range(bcfg.get('mc_samples_elbo', 5)):
        phi = phi_dist.rsample()
        expected_reward += (uncert * phi).mean()
    expected_reward = expected_reward / bcfg.get('mc_samples_elbo', 5)

    prior = torch.distributions.Beta(
        torch.full_like(alpha, bcfg.get('prior_alpha', 2.0)),
        torch.full_like(beta, bcfg.get('prior_beta', 5.0)),
    )
    kl = torch.distributions.kl.kl_divergence(phi_dist, prior).mean()
    entropy = phi_dist.entropy().mean()
    spread = torch.var(mean_phi)
    elbo = expected_reward - bcfg.get('lambda_kl', 0.01) * kl + bcfg.get('vi_entropy_weight', 0.02) * entropy + bcfg.get('var_weight', 0.20) * spread
    return -elbo


def compute_phi_loss(base_model: GLU_Bay_DD, out_logvar: Optional[torch.Tensor], y: torch.Tensor, cfg: Dict, epoch: int) -> torch.Tensor:
    if out_logvar is None or not cfg.get('Use_Adaptive_Selection', False):
        return torch.tensor(0.0, device=y.device)
    coords = y[0] if y.dim() == 3 else y
    uncert = torch.exp(out_logvar).detach().mean(dim=(0, 1, 3))
    uncert = normalize_01(uncert)
    anneal = min(epoch / max(cfg['bayesian_phi'].get('anneal_epochs', 100), 1), 1.0)
    return cfg['bayesian_phi'].get('lambda_elbo', 0.0) * anneal * compute_elbo(base_model, uncert, coords, cfg)


def maybe_downsample_batch(g_d: torch.Tensor, base_model: GLU_Bay_DD, cfg: Dict) -> torch.Tensor:
    if not cfg.get('BATCH_DOWNSAMPLE', False):
        return g_d
    b, t, n_x, n_call = g_d.shape
    ratio = torch.rand(1).item()
    ratio = ratio * (cfg['batch_downsample_max'] - cfg['batch_downsample_min']) + cfg['batch_downsample_min']
    ratio = float(max(1.0 / n_x, min(1.0, ratio)))
    if cfg.get('DOWNSAMPLE_LOGIC', 'random') == 'optimal' and hasattr(base_model, 'phi_mlp_1'):
        coords = g_d[:, 0, :, :2]
        log_ab = base_model.phi_mlp_1(coords)
        alpha = torch.exp(log_ab[..., 0]) + 1e-3
        beta = torch.exp(log_ab[..., 1]) + 1e-3
        phi = torch.clamp(alpha / (alpha + beta), min=1e-3, max=1 - 1e-3)
        num_keep = max(1, int(n_x * ratio))
        _, top_idx = torch.topk(phi, k=num_keep, dim=1)
        idx = top_idx.unsqueeze(1).unsqueeze(-1).expand(-1, t, -1, n_call)
        return torch.gather(g_d, dim=2, index=idx)
    num_keep = max(1, int(n_x * ratio))
    idx = torch.randperm(n_x, device=g_d.device)[:num_keep].view(1, 1, num_keep, 1).expand(b, t, -1, n_call)
    return torch.gather(g_d, dim=2, index=idx)


def run_epoch(model: nn.Module, base_model: GLU_Bay_DD, loader, optimizer: Optional[torch.optim.Optimizer], device: torch.device, cfg: Dict, channel_weights: torch.Tensor, epoch: int) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    metrics = {'loss': 0.0, 'mse': 0.0, 'obs': 0.0, 'traj': 0.0, 'phi': 0.0, 'nll': 0.0}
    n_batches = 0
    tf_prob = teacher_force_prob(epoch, cfg)
    nll_weight = cfg['nll_weight'] * min(epoch / max(cfg['nll_anneal_epochs'], 1), 1.0)
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            if len(batch) == 5:
                g_d, _g_dt, g_f, y, u = batch
            elif len(batch) == 6:
                g_d, _g_dt, g_f, _g_f_glb, y, u = batch
            else:
                raise ValueError(f"Unexpected batch structure of length {len(batch)}")
            g_d, g_f, y, u = [x.to(device) for x in (g_d, g_f, y, u)]
            if is_train:
                g_d = maybe_downsample_batch(g_d, base_model, cfg)
                optimizer.zero_grad(set_to_none=True)
            out, out_logvar, obs, traj, traj_logvar, _g_u_cls, g_u_mean_sens, g_u_logvar_sens = model(g_d, g_f, y, u, tf_prob)
            loss_mse, _ = weighted_channel_mse(out, g_f, channel_weights)
            loss_obs = sensor_supervision_loss(g_u_mean_sens, g_d, cfg, channel_weights)
            loss_traj = trajectory_loss(obs, traj, cfg)
            loss_nll = heteroscedastic_nll(out, out_logvar, g_f, g_u_logvar_sens, cfg, nll_weight)
            loss_phi = compute_phi_loss(base_model, out_logvar, y, cfg, epoch) if is_train else torch.tensor(0.0, device=device)
            # Keep the reported loss split into interpretable terms for monitoring and ablations.
            total = loss_mse + loss_obs + loss_traj + loss_phi + loss_nll
            if is_train:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                optimizer.step()
            metrics['loss'] += float(total.detach().cpu())
            metrics['mse'] += float(loss_mse.detach().cpu())
            metrics['obs'] += float(loss_obs.detach().cpu())
            metrics['traj'] += float(loss_traj.detach().cpu())
            metrics['phi'] += float(loss_phi.detach().cpu())
            metrics['nll'] += float(loss_nll.detach().cpu())
            n_batches += 1
    for k in metrics:
        metrics[k] /= max(n_batches, 1)
    return metrics


def maybe_save_importance_distributions(
    model: nn.Module,
    base_model: GLU_Bay_DD,
    loader,
    device: torch.device,
    cfg: Dict,
    epoch: int,
    net_name: str,
) -> None:
    if not cfg.get('save_imp_dists', False):
        return

    every = max(int(cfg.get('imp_dist_every', 1000)), 1)
    if epoch % every != 0:
        return

    out_dir = repo_dir / 'outputs' / 'eval' / f'Imp_{net_name}'
    out_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    first_uncert = None
    first_phi = None
    first_coords = None

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                g_d, _g_dt, g_f, y, u = batch
            elif len(batch) == 6:
                g_d, _g_dt, g_f, _g_f_glb, y, u = batch
            else:
                raise ValueError(f"Unexpected batch structure of length {len(batch)}")

            g_d, g_f, y, u = [x.to(device) for x in (g_d, g_f, y, u)]
            _out, out_logvar, _obs, _traj, _traj_logvar, _g_u_cls, _g_u_mean_sens, _g_u_logvar_sens = model(
                g_d, g_f, y, u, teacher_force_prob=0.0
            )

            if out_logvar is not None:
                first_uncert = torch.exp(out_logvar[0]).detach().cpu().numpy()
                first_coords = y[0].detach().cpu().numpy()

            if cfg.get('Use_Adaptive_Selection', False) and hasattr(base_model, 'phi_mlp_1'):
                full_coords = y[0]
                log_ab_1 = base_model.phi_mlp_1(full_coords)
                alpha_1 = torch.exp(log_ab_1[:, 0]) + 1e-3
                beta_1 = torch.exp(log_ab_1[:, 1]) + 1e-3
                phi_1 = torch.clamp(alpha_1 / (alpha_1 + beta_1), min=1e-3, max=1 - 1e-3)

                if (
                    cfg.get('Stage', 0) == 1
                    and cfg.get('bayesian_phi', {}).get('update_in_stage1', True)
                    and hasattr(base_model, 'phi_mlp_2')
                ):
                    log_ab_2 = base_model.phi_mlp_2(full_coords)
                    alpha_2 = torch.exp(log_ab_2[:, 0]) + 1e-3
                    beta_2 = torch.exp(log_ab_2[:, 1]) + 1e-3
                    phi_2 = torch.clamp(alpha_2 / (alpha_2 + beta_2), min=1e-3, max=1 - 1e-3)
                else:
                    phi_2 = torch.ones_like(phi_1)

                first_phi = (phi_1 * phi_2).detach().cpu().numpy()
            break

    if was_training:
        model.train(True)

    if first_uncert is None and first_phi is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    saved_any = False

    if first_uncert is not None:
        uncert_t0 = first_uncert[0, :, 0]
        sc = axes[0].scatter(first_coords[:, 0], first_coords[:, 1], c=uncert_t0, s=6, cmap='viridis')
        axes[0].set_title('Uncertainty field: first case')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(sc, ax=axes[0], shrink=0.8)
        np.savez_compressed(
            out_dir / f'epoch_{epoch:05d}_uncertainty.npz',
            values=first_uncert,
            coords=first_coords,
            epoch=epoch,
        )
        saved_any = True
    else:
        axes[0].set_visible(False)

    if first_phi is not None:
        phi_values = np.asarray(first_phi).reshape(-1)
        phi_coords = first_coords
        if phi_coords is not None and phi_coords.shape[0] == phi_values.shape[0]:
            sc = axes[1].scatter(phi_coords[:, 0], phi_coords[:, 1], c=phi_values, s=6, cmap='plasma', vmin=0.0, vmax=1.0)
        else:
            sc = axes[1].hist(phi_values, bins=64, range=(0.0, 1.0), color='tab:orange', alpha=0.85)
        axes[1].set_title('Importance field: first case')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        if phi_coords is not None and phi_coords.shape[0] == phi_values.shape[0]:
            plt.colorbar(sc, ax=axes[1], shrink=0.8)
        np.savez_compressed(
            out_dir / f'epoch_{epoch:05d}_importance.npz',
            values=phi_values,
            coords=phi_coords,
            epoch=epoch,
        )
        saved_any = True
    else:
        axes[1].set_visible(False)

    if saved_any:
        fig.tight_layout()
        fig.savefig(out_dir / f'epoch_{epoch:05d}_distributions.png', dpi=150)
    plt.close(fig)


def update_loss_history(
    history: list[Dict[str, float]],
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    out_dir: pathlib.Path,
) -> None:
    record = {'epoch': int(epoch)}
    for key, value in train_metrics.items():
        record[f'train_{key}'] = float(value)
    for key, value in val_metrics.items():
        record[f'val_{key}'] = float(value)
    history.append(record)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'loss_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    epochs = [row['epoch'] for row in history]
    train_loss = [max(row['train_mse'], 1e-12) for row in history]
    val_loss = [max(row['val_mse'], 1e-12) for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label='train mse', color='tab:blue', linewidth=2)
    ax.plot(epochs, val_loss, label='test mse', color='tab:orange', linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction MSE history')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'mse_loss_history.png', dpi=200)
    plt.close(fig)


def select_save_metric(train_metrics: Dict[str, float], val_metrics: Dict[str, float], cfg: Dict) -> tuple[str, float]:
    save_metric = str(cfg.get('save_metric', 'val_mse'))
    metric_map = {
        'train_loss': ('train_loss', train_metrics['loss']),
        'train_mse': ('train_mse', train_metrics['mse']),
        'val_loss': ('val_loss', val_metrics['loss']),
        'val_mse': ('val_mse', val_metrics['mse']),
        'test_loss': ('test_loss', val_metrics['loss']),
        'test_mse': ('test_mse', val_metrics['mse']),
    }
    return metric_map.get(save_metric, ('val_mse', val_metrics['mse']))


def train(cfg: Dict) -> None:
    seed_everything(cfg['seed'])
    device = torch.device(f"cuda:{cfg['device_ids'][0]}" if torch.cuda.is_available() else 'cpu')
    
    train_ld, val_ld, n_c, _ = make_data_loaders(cfg)
    model, net_name = build_model(cfg, n_c)
    maybe_load_stage0_weights(model, cfg)
    freeze_for_stage(model, cfg)
    if len(cfg['device_ids']) > 1 and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=cfg['device_ids'])
    
    model = model.to(device)
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    warmup_epochs = max(int(cfg['warmup_epochs']), 1)
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: min((e + 1) / warmup_epochs, 1.0))
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=500, min_lr=5e-6)
    
    channel_weights = get_channel_weights(cfg, n_c, device)
    save_dir = repo_dir / cfg['save_net_dir']
    save_dir.mkdir(parents=True, exist_ok=True)
    history_dir = repo_dir / 'outputs' / 'loss_history' / net_name
    best_metric = float('inf')
    
    patience_counter = 0
    loss_history: list[Dict[str, float]] = []
    for epoch in range(1, cfg['num_epochs'] + 1):
        train_metrics = run_epoch(model, base_model, train_ld, optimizer, device, cfg, channel_weights, epoch)
        val_metrics = run_epoch(model, base_model, val_ld, None, device, cfg, channel_weights, epoch)
        monitor_loader = train_ld if cfg.get('imp_dist_split', 'val').lower() == 'train' else val_ld
        maybe_save_importance_distributions(model, base_model, monitor_loader, device, cfg, epoch, net_name)
        if epoch <= warmup_epochs:
            warmup_sched.step()
        plateau_sched.step(train_metrics['mse'])
        if epoch % cfg['monitor_every'] == 0 or epoch == 1:
            update_loss_history(loss_history, epoch, train_metrics, val_metrics, history_dir)
            print(f"Epoch {epoch:05d} | train mse={train_metrics['mse']:.6f}, traj={train_metrics['traj']:.6f}, phi={train_metrics['phi']:.6f}, nll={train_metrics['nll']:.6f} | val mse={val_metrics['mse']:.6f}, traj={val_metrics['traj']:.6f}, nll={val_metrics['nll']:.6f}")
            if cfg.get('save_monitor_checkpoints', True):
                # Rolling checkpoint for recovery/resume; always overwritten in place.
                torch.save(
                    {'state_dict': base_model.state_dict(), 'epoch': epoch, 'best_metric': best_metric, 'config': cfg},
                    save_dir / f'Net_{net_name}_latest.pth'
                )

        metric_name, metric = select_save_metric(train_metrics, val_metrics, cfg)
        if metric < best_metric:
            best_metric = metric
            patience_counter = 0
            # "Best" checkpoint is selected every epoch, even though logs/plots are updated less often.
            torch.save(
                {'state_dict': base_model.state_dict(), 'epoch': epoch, 'best_metric': best_metric, 'best_metric_name': metric_name, 'config': cfg},
                save_dir / f'Net_{net_name}.pth'
            )
            print(f'Model Improving!...Saved best checkpoint for {net_name} at epoch {epoch} using {metric_name}={metric:.6f}\n')
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience_epochs']:
                print(f'Early stopping at epoch {epoch}; best {metric_name}={best_metric:.6f}')
                break


def main() -> None:
    args = parse_args()
    cfg = load_cfg(repo_dir / args.config)
    train(cfg)


if __name__ == '__main__':
    main()
