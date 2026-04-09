from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
import torch.nn.functional as F
import yaml

repo_dir = pathlib.Path(__file__).resolve().parent
src_dir = repo_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from src.dataloading import make_loaders, make_loaders_DSUS  
from src.models import GLU_Bay_DD, DomainAdaptiveEncoder, TemporalDecoderSoftmax, TemporalDecoderHierarchical, SoftDomainAdaptiveReconstructor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=pathlib.Path, required=True)
    p.add_argument('--checkpoint', type=pathlib.Path)
    p.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--frame_start', type=int, default=None, 
                   help='Optional starting frame index for deterministic single-sample evaluation.')
    p.add_argument('--num_sensors', type=int, default=None, 
                   help='Optional number of sensors to use for deterministic single-sample evaluation.')
    p.add_argument('--case_index_eval', type=int, default=0, 
                   help='Case index for deterministic single-sample evaluation.')
    p.add_argument('--sensor_seed', type=int, default=0, help='Seed used when selecting sensors for deterministic single-sample evaluation.')
    return p.parse_args()


def load_cfg(path: pathlib.Path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def resolve_config_path(path: pathlib.Path) -> pathlib.Path:
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            repo_dir / path,
            repo_dir / 'outputs' / 'configs' / path,
            repo_dir / 'configs' / path,
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Config not found. Tried: {", ".join(str(p) for p in candidates)}')


def infer_checkpoint_path(cfg: dict) -> pathlib.Path:
    net_name = f"GLU_id{cfg['case_index']}_st{cfg['Stage']}_num{cfg['Repeat_id']}"
    return repo_dir / cfg['save_net_dir'] / f'Net_{net_name}.pth'


def make_eval_out_dir(cfg: dict) -> pathlib.Path:
    dataset_name = pathlib.Path(cfg['data_h5']).parent.name
    time_tag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = repo_dir / 'outputs' / 'eval' / f'{dataset_name}_{time_tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def make_data_loaders(cfg: dict):
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


def build_model(cfg: dict, n_c: int) -> GLU_Bay_DD:
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

    reconstructor = SoftDomainAdaptiveReconstructor(
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

    return GLU_Bay_DD(
        cfg,
        encoder,
        temporal,
        reconstructor,
        delta_t=cfg['delta_t'],
        N_window=cfg['N_window'],
        stage=cfg['Stage'],
        use_adaptive_selection=cfg.get('Use_Adaptive_Selection', False),
        CalRecVar=cfg.get('CalRecVar', False),
        retain_cls=cfg.get('retain_cls', False),
        retain_lat=cfg.get('retain_lat', False),
        Use_imp_in_dyn=cfg.get('Use_imp_in_dyn', False),
    )


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num = torch.sqrt(torch.sum((pred - target) ** 2))
    den = torch.sqrt(torch.sum(target ** 2)) + 1e-8
    return num / den


def build_single_eval_batch(ds, frame_start: int | None, num_sensors: int | None, case_idx: int, sensor_seed: int):
    if case_idx < 0 or case_idx >= ds.B:
        raise ValueError(f'case_index_eval must be in [0, {ds.B - 1}], got {case_idx}')

    t0 = ds._rand_time_start() if frame_start is None else int(frame_start)
    max_t0 = ds.N_t - ds.T_full
    if ds.split == 'train':
        split_min = 0
        split_max = int(ds.Cut_Time * ds.idx_t_train) - ds.T_full + 1
    else:
        split_min = int(ds.Cut_Time * ds.idx_t_train) + 1
        split_max = int(ds.Cut_Time * ds.N_t) - ds.T_full
    if t0 < split_min or t0 > split_max or t0 < 0 or t0 > max_t0:
        raise ValueError(f'frame_start={t0} is invalid for split={ds.split}; valid range is [{split_min}, {split_max}]')

    n_sensors = ds.num_space_sample if num_sensors is None else int(num_sensors)
    if n_sensors <= 0 or n_sensors > len(ds.region_idx):
        raise ValueError(f'num_sensors must be in [1, {len(ds.region_idx)}], got {n_sensors}')

    u_case = ds.u[case_idx]
    U_case = ds.U[case_idx]
    u_case_glb = ds.u_glb[case_idx] if hasattr(ds, 'u_glb') else None

    t_full_idx = slice(t0, t0 + ds.T_full)
    t_obs_idx = slice(t0, t0 + ds.num_time_sample)

    if ds.Full_Field_DownS >= 1.0 - 1e-6:
        recon_idx = ds.recon_pool
    else:
        recon_gen = torch.Generator(device=ds.recon_pool.device if ds.recon_pool.is_cuda else 'cpu')
        recon_gen.manual_seed(sensor_seed + 1)
        recon_perm = torch.randperm(len(ds.recon_pool), generator=recon_gen, device=ds.recon_pool.device)
        recon_idx = ds.recon_pool[recon_perm[: ds.Num_recon_pts]]

    gen = torch.Generator(device=ds.region_idx.device if ds.region_idx.is_cuda else 'cpu')
    gen.manual_seed(sensor_seed)
    perm = torch.randperm(len(ds.region_idx), generator=gen, device=ds.region_idx.device)
    local_idx = perm[:n_sensors]
    obs_idx = ds.region_idx[local_idx]

    recon_idx, _ = recon_idx.sort()
    obs_idx, _ = obs_idx.sort()

    def gather(u, t_slice, s_idx):
        return u[t_slice][:, s_idx, :]

    G_full_u = gather(u_case, t_full_idx, recon_idx)
    G_down_t = gather(u_case, t_obs_idx, recon_idx)
    G_down_u = gather(u_case, t_obs_idx, obs_idx)
    G_full_glb = gather(u_case_glb, t_full_idx, recon_idx) if u_case_glb is not None else None

    Y_recon = ds.xy_exp[recon_idx]
    Y_obs = ds.xy_exp[obs_idx]
    t_obs = ds.t_exp[t_obs_idx]

    def build_cube(u_tensor, y, t_vec):
        T, Ns = u_tensor.shape[:2]
        y2 = y.unsqueeze(0).expand(T, -1, -1)
        t2 = t_vec.view(-1, 1, 1).expand(-1, Ns, 1)
        return torch.cat((y2, u_tensor, t2), dim=-1)

    G_down = build_cube(G_down_u, Y_obs, t_obs)
    if G_full_glb is not None:
        batch = (G_down.float(), G_down_t.float(), G_full_u.float(), G_full_glb.float(), Y_recon.float(), U_case.float())
    else:
        batch = (G_down.float(), G_down_t.float(), G_full_u.float(), Y_recon.float(), U_case.float())
    return tuple(x.unsqueeze(0) for x in batch)


def save_first_case_plot(
    y: torch.Tensor,
    gt: torch.Tensor,
    pred: torch.Tensor,
    out_dir: pathlib.Path,
    prefix: str,
    rel_l2_value: float,
    sensor_coords_scaled: torch.Tensor | None = None,
    PLOT_SENSORS: bool = True,
) -> None:
    coords = y.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    err_np = np.abs(pred_np - gt_np)
    sensor_coords_np = sensor_coords_scaled.detach().cpu().numpy() if sensor_coords_scaled is not None else None

    # first time step, first channel
    gt0 = gt_np[0, :, 0]
    pred0 = pred_np[0, :, 0]
    err0 = err_np[0, :, 0]

    triang = mtri.Triangulation(coords[:, 0], coords[:, 1])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    levels = 64
    for ax, arr, title in zip(axes, [gt0, pred0, err0], ['Ground truth', 'Prediction', 'Absolute error']):
        sc = ax.tricontourf(triang, arr, levels=levels, cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(sc, ax=ax, shrink=0.8)
    if sensor_coords_np is not None and PLOT_SENSORS:
        axes[0].scatter(
            sensor_coords_np[:, 0], sensor_coords_np[:, 1],
            s=12, c='lightgray', edgecolors='red', linewidths=1.0,
            marker='o', zorder=4, label='sensors',
        )
        axes[0].legend(frameon=False, loc='upper right', fontsize=8)
    fig.suptitle(f'Relative L2 Error: {rel_l2_value:.6f}', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f'{prefix}_first_case.png', dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    cfg = load_cfg(config_path)
    device = torch.device(args.device)
    checkpoint_path = repo_dir / args.checkpoint if args.checkpoint is not None else infer_checkpoint_path(cfg)

    train_ld, val_ld, n_c, _ = make_data_loaders(cfg)
    model = build_model(cfg, n_c).to(device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    out_dir = make_eval_out_dir(cfg)

    metrics = []
    single_sample_mode = args.frame_start is not None or args.num_sensors is not None or args.split == 'train'
    loader = train_ld if args.split == 'train' else val_ld

    with torch.no_grad():
        if single_sample_mode:
            batch_iter = [(0, build_single_eval_batch(loader.dataset, args.frame_start, args.num_sensors, args.case_index_eval, args.sensor_seed))]
        else:
            batch_iter = enumerate(loader)
        for i, batch in batch_iter:
            if len(batch) == 5:
                g_d, _g_dt, g_f, y, u = batch
            elif len(batch) == 6:
                g_d, _g_dt, g_f, _g_f_glb, y, u = batch
            else:
                raise ValueError(f'Unexpected batch length: {len(batch)}')
            g_d, g_f, y, u = [x.to(device) for x in (g_d, g_f, y, u)]
            out, out_logvar, obs, traj, traj_logvar, _g_u_cls, _g_u_mean_sens, _g_u_logvar_sens = model(g_d, g_f, y, u, teacher_force_prob=0.0)
            mse = F.mse_loss(out, g_f).item()
            l2 = rel_l2(out, g_f).item()
            metrics.append({'batch': i, 'mse': mse, 'rel_l2': l2})
            if i == 0:
                first_case_l2 = rel_l2(out[0], g_f[0]).item()
                save_first_case_plot(y[0], g_f[0], out[0], out_dir, 'val', first_case_l2, sensor_coords_scaled=g_d[0, 0, :, :2])

    summary = {
        'num_batches': len(metrics),
        'mean_mse': float(np.mean([m['mse'] for m in metrics])),
        'mean_rel_l2': float(np.mean([m['rel_l2'] for m in metrics])),
        'checkpoint': str(checkpoint_path),
        'config': str(config_path),
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({'summary': summary, 'batches': metrics}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
