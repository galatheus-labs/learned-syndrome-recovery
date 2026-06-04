"""Surface code decoder using ResNet3D.

Maps the syndrome (detector events) onto a 3D spatial grid matching the
surface code geometry, then uses our ResNet3D to predict the logical outcome.

For distance d with d rounds:
  - Detectors live on a ~(d+1) × (d+1) × (d+1) grid (sparse, checkerboard)
  - We embed them into a dense 3D volume and feed to ResNet3D
  - Output: single logit for logical error prediction

Training uses random error rates (uniform) so one model handles all noise
levels — same trick as our random DCT level training for variable-rate
compression.

Usage:
    python surface_code_resnet.py --distance 5 --noise circuit
    python surface_code_resnet.py --distance 5 --noise circuit --eval-only --ckpt ckpts/qec_d5.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

import stim
import pymatching


def get_detector_grid_mapping(circuit):
    """Map detector indices to 3D grid coordinates."""
    coords = circuit.get_detector_coordinates()
    coords_arr = np.array([coords[i] for i in range(len(coords))])

    mappings = []
    sizes = []
    for dim in range(coords_arr.shape[1]):
        unique = sorted(set(coords_arr[:, dim]))
        val_to_idx = {v: i for i, v in enumerate(unique)}
        mappings.append(val_to_idx)
        sizes.append(len(unique))

    det_to_grid = {}
    for det_idx in range(len(coords_arr)):
        x = mappings[0][coords_arr[det_idx, 0]]
        y = mappings[1][coords_arr[det_idx, 1]]
        z = mappings[2][coords_arr[det_idx, 2]]
        det_to_grid[det_idx] = (x, y, z)

    return tuple(sizes), det_to_grid


def detectors_to_volume(det_events, grid_size, det_to_grid):
    """Convert flat detector events (B, n_det) to 3D volumes (B, 1, Dx, Dy, Dz)."""
    B = det_events.shape[0]
    Dx, Dy, Dz = grid_size
    volume = torch.zeros(B, 1, Dx, Dy, Dz)
    for det_idx, (x, y, z) in det_to_grid.items():
        volume[:, 0, x, y, z] = det_events[:, det_idx]
    return volume


def d4_augment_all(vols):
    """Pre-compute the LABEL-PRESERVING symmetries of a planar surface code.

    The planar surface code is NOT D4-symmetric: two opposite boundaries
    are X-type and the other two are Z-type. Of the 8 D4 elements, only
    the Z2 x Z2 subgroup that preserves boundary types and the logical
    observable is valid:

      - identity
      - 180° rotation                 (preserves both boundary types)
      - reflection across x-axis      (preserves boundary types pairwise)
      - reflection across y-axis      (preserves boundary types pairwise)

    The other 4 (90°, 270°, diagonal reflections) swap X and Z boundaries
    and therefore swap the X̄/Z̄ logical operators. Applying them with the
    same label injects contradictory training data and floors the model.

    Returns 4x the data with the same labels.

    Args:
        vols: (N, C, Dx, Dy, Dz) tensor

    Returns:
        (4*N, C, Dx, Dy, Dz) tensor — 4 valid transforms concatenated
    """
    transforms = [
        vols,                          # identity
        torch.rot90(vols, 2, [2, 3]),  # 180°
        torch.flip(vols, [2]),         # reflect x
        torch.flip(vols, [3]),         # reflect y
    ]
    return torch.cat(transforms, dim=0)


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Focal loss for class-imbalanced binary classification.

    At d≥9, logical errors are rare events (e.g., 0.06% at p=0.003).
    Standard BCE treats all samples equally, wasting capacity on the
    easy majority. Focal loss down-weights well-classified examples
    and focuses on hard/rare cases.

    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    # Alpha weighting: alpha for positive class, (1-alpha) for negative
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * focal_weight * bce).mean()


class MixedErrorRateDataset(Dataset):
    """Dataset that samples from multiple error rates.

    Each __getitem__ picks a random error rate's data, providing the
    detector volume + error rate as conditioning + observable label.

    Supports oversampling of positive (error) examples.
    D4 augmentation is done up front via d4_augment_all() before
    constructing this dataset (zero per-batch overhead).
    """

    def __init__(self, volumes_by_rate, labels_by_rate, error_rates,
                 oversample_positive=1):
        self.volumes = volumes_by_rate   # list of (N, C, Dx, Dy, Dz) tensors
        self.labels = labels_by_rate     # list of (N,) tensors
        self.error_rates = error_rates

        # Build index: list of (rate_idx, sample_idx)
        self.index = []
        for rate_idx in range(len(error_rates)):
            n = len(self.volumes[rate_idx])
            for sample_idx in range(n):
                self.index.append((rate_idx, sample_idx))
                if oversample_positive > 1 and self.labels[rate_idx][sample_idx] > 0.5:
                    for _ in range(oversample_positive - 1):
                        self.index.append((rate_idx, sample_idx))

        self.total = len(self.index)
        n_pos = sum(1 for r, s in self.index if self.labels[r][s] > 0.5)
        print(f"  Dataset: {self.total} samples "
              f"({n_pos} positive, {self.total - n_pos} negative)",
              flush=True)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        rate_idx, sample_idx = self.index[idx]
        vol = self.volumes[rate_idx][sample_idx]
        label = self.labels[rate_idx][sample_idx]
        p = self.error_rates[rate_idx]
        return vol, label, torch.tensor(p, dtype=torch.float32)


class ResNetDecoder(nn.Module):
    """3D ResNet for syndrome → logical outcome prediction.

    Includes error rate conditioning — the model knows what noise level
    it's decoding, analogous to the timestep embedding in diffusion.
    """

    def __init__(self, grid_size, width=64, depth=8, cond_dim=32, in_channels=1,
                 zero_init_residual=False, block_type="classic", se_ratio=8):
        super().__init__()
        self.cond_dim = cond_dim
        self.block_type = block_type

        # Error rate embedding
        self.rate_mlp = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.input_conv = nn.Conv3d(in_channels, width, 3, padding=1)

        block_cls = {"classic": CondResBlock, "lc0": CondResBlockLC0}[block_type]
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            if block_type == "lc0":
                self.blocks.append(block_cls(width, cond_dim, se_ratio=se_ratio,
                                             zero_init=zero_init_residual))
            else:
                self.blocks.append(block_cls(width, cond_dim))

        # For pre-activation blocks we need a final norm+activation before the head.
        self.post_norm = nn.GroupNorm(8, width) if block_type == "lc0" else None

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

        if zero_init_residual and block_type == "classic":
            # Zero the final GroupNorm γ in each residual block so the tower
            # starts as identity — avoids variance compounding at high depth.
            for block in self.blocks:
                nn.init.zeros_(block.gn2.weight)
        # LC0 blocks handle zero-init internally via a ReZero-style scalar.

    def forward(self, x, error_rate=None):
        h = self.input_conv(x)
        if self.block_type == "classic":
            h = F.silu(h)

        if error_rate is not None:
            cond = self.rate_mlp(error_rate.view(-1, 1))
        else:
            cond = None

        for block in self.blocks:
            h = block(h, cond)

        if self.post_norm is not None:
            h = F.silu(self.post_norm(h))

        h = self.pool(h).flatten(1)
        return self.head(h).squeeze(-1)


class CondResBlock(nn.Module):
    """Residual block with optional conditioning (error rate)."""

    def __init__(self, channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels)
        self.gn2 = nn.GroupNorm(8, channels)
        self.cond_proj = nn.Linear(cond_dim, channels)

    def forward(self, x, cond=None):
        h = F.silu(self.gn1(self.conv1(x)))
        if cond is not None:
            h = h + self.cond_proj(cond).view(cond.size(0), -1, 1, 1, 1)
        h = self.gn2(self.conv2(h))
        return F.silu(x + h)


class CondResBlockLC0(nn.Module):
    """LC0-style residual block: pre-activation, dual-output SE, FiLM cond.

    Reference: Leela Chess Zero residual tower.
      - Pre-activation ordering (GN -> SiLU -> conv) gives a clean identity
        skip, which trains better at high depth.
      - SE block has a dual output (scale + bias per channel). The bias is
        LC0's innovation — standard SE only multiplies.
      - Error-rate conditioning is injected as per-channel FiLM (scale+bias),
        not a scalar additive shift.
      - Optional ReZero scalar (learnable, init=0) lets the block start as
        identity, mitigating variance blow-up at high depth.
    """

    def __init__(self, channels, cond_dim, se_ratio=8, zero_init=False):
        super().__init__()
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

        mid = max(channels // se_ratio, 8)
        self.se_fc1 = nn.Linear(channels, mid)
        self.se_fc2 = nn.Linear(mid, 2 * channels)

        self.cond_proj = nn.Linear(cond_dim, 2 * channels)

        # ReZero: scalar gate on the residual branch, init=0 → identity block.
        self.rezero = nn.Parameter(torch.zeros(1)) if zero_init else None

    def forward(self, x, cond=None):
        # Pre-activation residual branch
        h = self.conv1(F.silu(self.gn1(x)))
        h = self.conv2(F.silu(self.gn2(h)))

        # LC0 SE: per-channel scale (sigmoid-gated) + additive bias
        s = h.mean(dim=(-3, -2, -1))                     # (B, C)
        s = self.se_fc2(F.silu(self.se_fc1(s)))          # (B, 2C)
        se_scale, se_bias = s.chunk(2, dim=1)
        h = h * torch.sigmoid(se_scale).view(*se_scale.shape, 1, 1, 1) \
              + se_bias.view(*se_bias.shape, 1, 1, 1)

        # FiLM from error-rate conditioning (per-channel scale + bias)
        if cond is not None:
            c_scale, c_bias = self.cond_proj(cond).chunk(2, dim=1)
            h = h * (1 + c_scale).view(*c_scale.shape, 1, 1, 1) \
                  + c_bias.view(*c_bias.shape, 1, 1, 1)

        if self.rezero is not None:
            h = h * self.rezero

        return x + h


def compute_local_syndrome_density(det_np, grid_size, det_to_grid, kernel_size=3):
    """Compute local detector density as a spatial prior channel.

    For each grid cell, count how many detectors fired in a local
    neighborhood. This gives the ResNet a "where are errors concentrated"
    prior — analogous to the surface-patch constraints in the compression
    paper. Regions with many nearby fired detectors are likely to contain
    correlated errors that MWPM mishandles.

    This is a content-adaptive spatial prior derived from the raw
    syndrome, requiring zero extra bits — both encoder and decoder
    can compute it from the detector events.

    Args:
        det_np: (N, n_detectors) bool array
        grid_size: (Dx, Dy, Dz) tuple
        det_to_grid: dict mapping detector_idx -> (x, y, z)
        kernel_size: neighborhood size for density computation

    Returns: (N, 1, Dx, Dy, Dz) float tensor — local density volume
    """
    det_vol = detectors_to_volume(
        torch.from_numpy(det_np.astype(np.float32)), grid_size, det_to_grid)

    # Apply 3D average pooling to compute local density, then upsample back
    pad = kernel_size // 2
    density = F.avg_pool3d(det_vol, kernel_size=kernel_size, stride=1, padding=pad)

    return density


def generate_data(distance, error_rate, n_samples, noise='circuit', sampler_seed=None):
    """Generate detector events and observable flips."""
    if noise == 'circuit':
        circuit = stim.Circuit.generated(
            'surface_code:rotated_memory_z',
            rounds=distance, distance=distance,
            after_clifford_depolarization=error_rate,
            after_reset_flip_probability=error_rate,
            before_measure_flip_probability=error_rate,
            before_round_data_depolarization=error_rate,
        )
    elif noise == 'phenomenological':
        circuit = stim.Circuit.generated(
            'surface_code:rotated_memory_z',
            rounds=distance, distance=distance,
            before_round_data_depolarization=error_rate,
            before_measure_flip_probability=error_rate,
        )
    else:
        circuit = stim.Circuit.generated(
            'surface_code:rotated_memory_z',
            rounds=1, distance=distance,
            before_round_data_depolarization=error_rate,
        )

    sampler = circuit.compile_detector_sampler(seed=sampler_seed)
    det, obs = sampler.sample(shots=n_samples, separate_observables=True)
    return det, obs, circuit


def _log(msg, log_file=None):
    """Print and optionally write to log file (bypasses tee buffering)."""
    print(msg, flush=True)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(msg + '\n')


def train_mixed(distance, noise='circuit', n_per_rate=200000, n_val=1000000,
                epochs=40, width=64, depth=8, batch_size=2048,
                train_rates=None, eval_rates=None, save_dir=None, seed=42,
                local_prior=False, augment=False, use_focal_loss=False,
                oversample_positive=1, log_file=None,
                block_type='classic', zero_init_residual=False):
    """Train one model on mixed error rates, evaluate on each."""

    if train_rates is None:
        train_rates = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
    if eval_rates is None:
        eval_rates = [0.001, 0.003, 0.005, 0.007]

    # Reproducibility: seed all RNGs (numpy, torch, stim)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}, seed: {seed}', flush=True)
    print(f'Training on {len(train_rates)} error rates: {train_rates}', flush=True)

    # Generate training data for each error rate
    # Use one circuit to get grid mapping (same for all rates at same d)
    _, _, ref_circuit = generate_data(distance, train_rates[0], 1, noise)
    grid_size, det_to_grid = get_detector_grid_mapping(ref_circuit)
    n_det = ref_circuit.num_detectors
    print(f'd={distance}, noise={noise}, detectors={n_det}, grid={grid_size}', flush=True)

    print(f'Generating {n_per_rate} samples × {len(train_rates)} rates...', flush=True)
    train_volumes = []
    train_labels = []
    for i, p in enumerate(train_rates):
        # Distinct stim sampler seed per rate, derived from the master seed
        det, obs, _ = generate_data(distance, p, n_per_rate, noise,
                                    sampler_seed=seed * 1000 + i)
        vol = detectors_to_volume(
            torch.from_numpy(det.astype(np.float32)), grid_size, det_to_grid)
        if local_prior:
            density = compute_local_syndrome_density(det, grid_size, det_to_grid)
            vol = torch.cat([vol, density], dim=1)  # (N, 2, Dx, Dy, Dz)
        train_volumes.append(vol)  # keep on CPU
        train_labels.append(torch.from_numpy(obs.astype(np.float32)).squeeze(-1))
        print(f'  p={p}: obs_flip_rate={obs.mean():.4f}', flush=True)

    # Generate validation data per eval rate
    print(f'Generating {n_val} val samples × {len(eval_rates)} rates...', flush=True)
    val_data = {}
    # Use a different (deterministic) seed range for val so train/val don't overlap
    for i, p in enumerate(eval_rates):
        det, obs, circuit = generate_data(distance, p, n_val, noise,
                                          sampler_seed=seed * 1000 + 500 + i)
        vol = detectors_to_volume(
            torch.from_numpy(det.astype(np.float32)), grid_size, det_to_grid)
        if local_prior:
            density = compute_local_syndrome_density(det, grid_size, det_to_grid)
            vol = torch.cat([vol, density], dim=1)
        # MWPM baseline
        dem = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        mwpm_pred = matcher.decode_batch(det)
        mwpm_err = (mwpm_pred != obs).sum() / n_val
        val_data[p] = {
            'vol': vol,
            'obs': obs.squeeze(-1),
            'mwpm': float(mwpm_err),
        }
        print(f'  p={p}: MWPM={mwpm_err:.6f}', flush=True)

    # D4 augmentation: expand training data with the label-preserving subgroup
    if augment:
        print('Applying D4 augmentation...', end=' ', flush=True)
        aug_volumes = []
        aug_labels = []
        for vol, lab in zip(train_volumes, train_labels):
            base_count = len(lab)
            aug_vol = d4_augment_all(vol)
            aug_factor, remainder = divmod(len(aug_vol), base_count)
            if remainder != 0:
                raise ValueError(
                    f'D4 augmentation produced {len(aug_vol)} volumes for '
                    f'{base_count} labels'
                )
            aug_lab = lab.repeat(aug_factor)
            aug_volumes.append(aug_vol)
            aug_labels.append(aug_lab)
        train_volumes = aug_volumes
        train_labels = aug_labels
        n_total = sum(len(v) for v in train_volumes)
        print(f'{n_total:,} samples', flush=True)

    # Build mixed dataset
    dataset = MixedErrorRateDataset(train_volumes, train_labels, train_rates,
                                    oversample_positive=oversample_positive)
    # num_workers=0: pre-augmented volumes are ~11GB; forked workers would
    # duplicate them (69GB total for 4 workers). Single-process loading
    # is fine since the data is already in memory as tensors.
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    if use_focal_loss:
        print(f'Focal loss: enabled (alpha=0.25, gamma=2.0)', flush=True)
    if oversample_positive > 1:
        print(f'Positive oversampling: {oversample_positive}x', flush=True)

    # Model
    in_ch = 2 if local_prior else 1
    model = ResNetDecoder(grid_size, width=width, depth=depth, in_channels=in_ch,
                          block_type=block_type,
                          zero_init_residual=zero_init_residual).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'ResNet3D: {params:,} params (w={width}, d={depth}, block={block_type}, zi={zero_init_residual})',
          flush=True)

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = device == 'cuda' and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    if use_amp:
        print(f'Using bf16 AMP', flush=True)

    best_results = {p: 1.0 for p in eval_rates}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        for vol, label, p_val in train_loader:
            vol, label, p_val = vol.to(device), label.to(device), p_val.to(device)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                logits = model(vol, p_val)
                if use_focal_loss:
                    loss = focal_loss(logits, label)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, label)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Evaluate
        model.eval()
        if (epoch + 1) % 1 == 0:
            line = f'  Epoch {epoch+1}: loss={total_loss/n_batches:.4f}'
            with torch.no_grad():
                for p in eval_rates:
                    vd = val_data[p]
                    all_preds = []
                    for i in range(0, len(vd['vol']), batch_size):
                        batch_vol = vd['vol'][i:i+batch_size].to(device)
                        batch_p = torch.full((len(batch_vol),), p, device=device)
                        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                            logits = model(batch_vol, batch_p)
                        all_preds.append((logits > 0).cpu().numpy().astype(np.uint8))
                    pred = np.concatenate(all_preds)
                    neural_err = (pred != vd['obs']).sum() / n_val
                    if neural_err < best_results[p]:
                        best_results[p] = float(neural_err)
                    line += f'  p={p}:{neural_err:.5f}'
            _log(line, log_file)

            if save_dir is not None:
                from pathlib import Path
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch + 1,
                    'grid_size': grid_size,
                    'det_to_grid': det_to_grid,
                    'width': width,
                    'depth': depth,
                    'distance': distance,
                    'noise': noise,
                    'best_results': best_results,
                }, f'{save_dir}/qec_d{distance}_epoch{epoch+1}.pt')

    # Final summary
    print(f'\n===== Summary: {noise} noise, d={distance} (mixed training) =====',
          flush=True)
    print(f'{"p":>8} {"MWPM":>10} {"ResNet":>10} {"Diff":>10}', flush=True)
    for p in eval_rates:
        m = val_data[p]['mwpm']
        n = best_results[p]
        diff = (m - n) / m * 100 if m > 0 else 0
        print(f'{p:>8.4f} {m:>10.6f} {n:>10.6f} {diff:>+9.1f}%', flush=True)

    return val_data, best_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--noise", type=str, default="circuit",
                        choices=["code_capacity", "phenomenological", "circuit"])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--n-per-rate", type=int, default=200000)
    parser.add_argument("--n-val", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--save-dir", type=str, default="ckpts/qec")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (Stim, numpy, torch)")
    parser.add_argument("--local-prior", action="store_true",
                        help="Add local syndrome density as 2nd input channel")
    parser.add_argument("--augment", action="store_true",
                        help="Label-preserving D4 subgroup augmentation (4x effective data)")
    parser.add_argument("--focal-loss", action="store_true",
                        help="Use focal loss for class-imbalanced training")
    parser.add_argument("--oversample", type=int, default=1,
                        help="Oversample positive (error) examples N times")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Write epoch results directly to this file (bypasses tee)")
    parser.add_argument("--block-type", type=str, default="classic",
                        choices=["classic", "lc0"],
                        help="Residual block variant")
    parser.add_argument("--zero-init-residual", action="store_true",
                        help="Zero-init residual branch (identity-at-init)")
    args = parser.parse_args()

    train_mixed(
        args.distance, noise=args.noise,
        n_per_rate=args.n_per_rate, n_val=args.n_val,
        epochs=args.epochs,
        width=args.width, depth=args.depth, batch_size=args.batch_size,
        save_dir=args.save_dir, seed=args.seed,
        local_prior=args.local_prior,
        augment=args.augment,
        use_focal_loss=args.focal_loss,
        oversample_positive=args.oversample,
        log_file=args.log_file,
        block_type=args.block_type,
        zero_init_residual=args.zero_init_residual)
