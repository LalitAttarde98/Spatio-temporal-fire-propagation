"""
Improved Fire Propagation Dataset
===================================
Changes from baseline
─────────────────────
Features
  + Signed Distance Field  (SDF)  of fire front — richer spatial context than binary mask
  + Rate-of-Spread magnitude |V_normal| — direct physical quantity
  + Front curvature  κ = div(∇φ / |∇φ|) — affects spread at curved fronts (Huygens)
  + Wind-slope interaction  u·cos(α)  — dominant RoS driver made explicit

Augmentation  (AugTransform)
  ✓ Lateral flip  dims=[-1]  kept — width=32 is homogeneous
  ✗ Height flip  removed — fire propagates 0→113 so this would invert physics
  + Lateral circular roll  — full translational invariance in width direction
  + Wind-speed scaling  [0.85, 1.15]  with consistent recomputation of derived features
  + Slope jitter  ±2°  with consistent recomputation
  + Refined per-feature noise levels matched to the expanded feature set

Dataset / Sampling
  ✓ Precomputed & cached features — expensive numpy ops run once per simulation
  ✓ Oversampling bug fixed — shadowed variable `i` renamed to `rep`
  + Activity-weighted window sampling — windows with active fire transitions are
    oversampled proportionally to the number of newly ignited pixels, so the
    model sees more examples of the interesting propagation dynamics

Feature channel layout (13 total)
  0  wind_speed
  1  terrain_slope
  2  temperatures
  3  velocities (ustar)
  4  vertical_velocity
  5  horizontal_velocity
  6  ff_cumsum
  7  diff_temp
  8  diff_vel
  9  ff_vx  (level-set x-velocity)
 10  ff_vy  (level-set y-velocity)
 11  sdf                ← new: signed distance to fire front
 12  ros_magnitude      ← new: |V_normal| rate of spread
 13  curvature          ← new: ∇·(∇φ/|∇φ|)
 14  wind_slope_interact← new: u·cos(α)
 15  firefronts         ← prediction target (always last)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as Tv
from scipy.ndimage import gaussian_filter, distance_transform_edt


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_sdf(firefronts: np.ndarray) -> np.ndarray:
    """
    Signed Distance Field for each frame's binary fire-front mask.

    Inside the fire  → negative distance  (already burned)
    Outside the fire → positive distance  (unburned)
    Clipped to [-30, 30] and normalised by 30 for scale invariance.

    firefronts: (T, H, W)  binary  float32
    returns:    (T, H, W)  float32
    """
    sdf = np.zeros_like(firefronts)
    for t in range(firefronts.shape[0]):
        mask = firefronts[t] > 0.5
        if mask.any() and not mask.all():
            dist_out = distance_transform_edt(~mask).astype(np.float32)
            dist_in  = distance_transform_edt( mask).astype(np.float32)
            sdf[t]   = dist_out - dist_in
        elif mask.all():
            sdf[t] = -distance_transform_edt(np.ones_like(mask)).astype(np.float32)
        # else all zeros — no fire yet, leave as 0
    return np.clip(sdf / 30.0, -1.0, 1.0)


def compute_curvature(phi: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Mean curvature of the level-set  κ = div(∇φ / |∇φ|).

    High |κ| → tightly curved front (fire accelerates at tips).
    phi:    (T, H, W)  smoothed level-set field
    returns (T, H, W)  float32, clipped to [-1, 1]
    """
    gy, gx = np.gradient(phi, axis=(1, 2))
    g_mag  = np.sqrt(gx**2 + gy**2 + eps)
    nx, ny = gx / g_mag, gy / g_mag
    # curvature = ∂nx/∂x + ∂ny/∂y
    _, dnx_dx = np.gradient(nx, axis=(1, 2))   # np.gradient returns (dy, dx)
    dny_dy, _ = np.gradient(ny, axis=(1, 2))
    kappa = (dnx_dx + dny_dy).astype(np.float32)
    return np.clip(kappa, -1.0, 1.0)


def firefront_features(firefronts: np.ndarray):
    """
    Derives physical features from the binary fire-front sequence.

    firefronts: (T, H, W)  float32

    Returns
    -------
    vx, vy       level-set normal velocity components
    ff_cumsum    cumulative burned area
    sdf          signed distance field
    ros_mag      rate-of-spread magnitude |V_normal|
    curvature    front curvature κ
    """
    smooth = gaussian_filter(firefronts.astype(np.float32), sigma=(0, 0.5, 0.5))

    dphi_dt = np.diff(smooth, axis=0, prepend=smooth[0:1])
    gy, gx  = np.gradient(smooth, axis=(1, 2))
    grad2   = gx**2 + gy**2
    mask    = grad2 > 1e-6

    vx = np.zeros_like(dphi_dt)
    vy = np.zeros_like(dphi_dt)
    vx[mask] = -dphi_dt[mask] * (gx[mask] / grad2[mask])
    vy[mask] = -dphi_dt[mask] * (gy[mask] / grad2[mask])

    ros_mag   = np.sqrt(vx**2 + vy**2).astype(np.float32)
    ff_cumsum = np.cumsum(firefronts, axis=0).astype(np.float32)
    sdf       = compute_sdf(firefronts)
    curvature = compute_curvature(smooth)

    return vx, vy, ff_cumsum, sdf, ros_mag, curvature


def build_features(
    wind_speed:     np.ndarray,   # (T, H, W)
    terrain_slope:  np.ndarray,   # (T, H, W)  in radians
    temperatures:   np.ndarray,
    velocities:     np.ndarray,
    firefronts:     np.ndarray,
) -> np.ndarray:
    """
    Assemble the full (T, C, H, W) feature tensor from raw simulation fields.
    Reusable so augmentation can call it after modifying physical parameters.
    """
    vert_vel  = (velocities - wind_speed * np.cos(terrain_slope)) \
                / np.sin(terrain_slope + 1e-4)
    horiz_vel = wind_speed * np.cos(terrain_slope)

    vx, vy, ff_cumsum, sdf, ros_mag, curvature = firefront_features(firefronts)

    diff_temp = np.diff(temperatures, axis=0, prepend=temperatures[0:1])
    diff_vel  = np.diff(velocities,   axis=0, prepend=velocities[0:1])

    wind_slope = wind_speed * np.cos(terrain_slope)   # u·cos(α), dominant RoS driver

    features = np.stack([
        wind_speed,        #  0
        terrain_slope,     #  1
        temperatures,      #  2
        velocities,        #  3
        vert_vel,          #  4
        horiz_vel,         #  5
        ff_cumsum,         #  6
        diff_temp,         #  7
        diff_vel,          #  8
        vx,                #  9
        vy,                # 10
        sdf,               # 11  ← new
        ros_mag,           # 12  ← new
        curvature,         # 13  ← new
        wind_slope,        # 14  ← new
        firefronts,        # 15  ← prediction target, always last
    ], axis=1)  # (T, C, H, W)

    return features.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation transform
# ─────────────────────────────────────────────────────────────────────────────

# Channel indices that stay constant (physical constants / identifiers)
_NO_NOISE = {0, 1, 6, 11, 15}     # wind_speed, slope, cumsum, sdf, firefronts

# Per-channel noise std (0 = no noise); length = 16
_NOISE_STD = torch.tensor([
    0.00,   #  0 wind_speed      — perturbed in physical-param aug instead
    0.00,   #  1 terrain_slope   — same
    1.00,   #  2 temperatures
    0.10,   #  3 velocities
    0.10,   #  4 vertical_velocity
    0.10,   #  5 horizontal_velocity
    0.00,   #  6 ff_cumsum
    0.05,   #  7 diff_temp
    0.05,   #  8 diff_vel
    0.05,   #  9 ff_vx
    0.05,   # 10 ff_vy
    0.00,   # 11 sdf
    0.02,   # 12 ros_magnitude
    0.01,   # 13 curvature
    0.00,   # 14 wind_slope_interact
    0.00,   # 15 firefronts
]).view(1, -1, 1, 1)


class AugTransform:
    """
    Augmentations for a (T, C, H, W) float32 tensor where
      H = 113  (slope / propagation direction — do NOT flip)
      W = 32   (lateral direction — homogeneous, safe to flip / roll)

    Augmentations applied
    ─────────────────────
    p=0.5   lateral flip          dim=-1 (W)
    p=0.7   lateral circular roll random shift ∈ [0, W)
    p=0.3   gaussian blur on non-firefront channels
    always  additive Gaussian noise at per-channel levels
    """

    def __init__(self):
        self.noise_std = _NOISE_STD   # (1, C, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, C, H, W)  float32"""

        # ── Lateral flip (W dim) ─────────────────────────────────────────────
        # Safe because lateral direction has no preferred orientation.
        # Do NOT flip H (slope / propagation axis).
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[-1])

        # ── Lateral circular roll ────────────────────────────────────────────
        # Equivalent to observing the same fire at a different lateral position.
        if torch.rand(1) < 0.7:
            shift = torch.randint(0, x.shape[-1], (1,)).item()
            x = torch.roll(x, shifts=shift, dims=-1)

        # ── Gaussian blur on dynamic channels (not fronts / sdf) ─────────────
        if torch.rand(1) < 0.3:
            blur_ch = [c for c in range(x.shape[1]) if c not in {11, 15}]
            x[:, blur_ch] = Tv.gaussian_blur(x[:, blur_ch], kernel_size=(3, 3))

        # ── Additive Gaussian noise ──────────────────────────────────────────
        noise = torch.randn_like(x) * self.noise_std.to(x.device)
        x = x + noise

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Physical-parameter augmentation (applied before feature engineering)
# ─────────────────────────────────────────────────────────────────────────────

class PhysicalAug:
    """
    Perturbs raw physical parameters and recomputes ALL derived features from
    scratch, keeping the feature tensor internally consistent.

    This is qualitatively different from noise injection: it synthesises a new
    physically valid simulation variant, effectively multiplying the dataset.

    Parameters
    ──────────
    wind_scale_range   multiplicative range for wind speed  (U ← U·s)
    slope_jitter_deg   additive range for terrain slope in degrees
    p_wind             probability of applying wind scaling
    p_slope            probability of applying slope jitter
    """

    def __init__(
        self,
        wind_scale_range: tuple = (0.85, 1.15),
        slope_jitter_deg: float = 2.0,
        p_wind:           float = 0.5,
        p_slope:          float = 0.4,
    ):
        self.wind_lo, self.wind_hi = wind_scale_range
        self.slope_jitter = slope_jitter_deg * (np.pi / 180.0)
        self.p_wind  = p_wind
        self.p_slope = p_slope

    def __call__(
        self,
        wind_speed:    np.ndarray,
        terrain_slope: np.ndarray,
        temperatures:  np.ndarray,
        velocities:    np.ndarray,
        firefronts:    np.ndarray,
    ) -> np.ndarray:
        """Returns augmented (T, C, H, W) feature tensor."""
        ws = wind_speed.copy()
        ts = terrain_slope.copy()

        if np.random.rand() < self.p_wind:
            scale = np.random.uniform(self.wind_lo, self.wind_hi)
            ws = ws * scale

        if np.random.rand() < self.p_slope:
            jitter = np.random.uniform(-self.slope_jitter, self.slope_jitter)
            ts = np.clip(ts + jitter, 0.0, np.pi / 2 - 1e-3)

        return build_features(ws, ts, temperatures, velocities, firefronts)


# ─────────────────────────────────────────────────────────────────────────────
# Base simulation dataset  (loads & caches one simulation per index)
# ─────────────────────────────────────────────────────────────────────────────

class FireDataset(Dataset):
    """
    Loads raw simulation files and builds the feature tensor.

    Features are **cached in memory** after the first access so that the
    expensive numpy ops (SDF, curvature, level-set velocities) run only once
    per simulation regardless of how many windows are sampled from it.

    For validation, no augmentation is applied.
    """

    def __init__(self, data_cfg: dict, Enable_validation: bool = False):
        self.data_dir         = data_cfg['data_dir']
        self.enable_aug       = data_cfg.get('enable_augmentation', True) \
                                and not Enable_validation
        self.Enable_validation = Enable_validation

        csv_path = data_cfg['val_csv'] if Enable_validation else data_cfg['data_csv']
        raw = np.genfromtxt(csv_path, delimiter=',', dtype=None,
                            names=True, encoding='utf-8')
        self.data = np.array([raw.item()], dtype=raw.dtype) \
                    if raw.ndim == 0 else raw

        self.aug_transform = AugTransform()
        self.phys_aug      = PhysicalAug()

        # Cache: index → (T, C, H, W) float32 ndarray
        self._cache: dict[int, np.ndarray] = {}

    def _load_raw(self, idx: int):
        """Load raw fields from disk for simulation idx."""
        m     = self.data[idx]
        shape = (m['Nt'], m['Nx'], m['Ny'])

        wind_speed    = np.full(shape, m['u'],     dtype='float32')
        terrain_slope = np.full(shape, m['alpha'], dtype='float32') \
                        * (np.pi / 180.0)

        temperatures = np.fromfile(
            os.path.join(self.data_dir, m['theta_filename']),
            dtype='<f4').reshape(shape).astype('float32')

        velocities = np.fromfile(
            os.path.join(self.data_dir, m['ustar_filename']),
            dtype='<f4').reshape(shape).astype('float32')

        firefronts = np.fromfile(
            os.path.join(self.data_dir, m['xi_filename']),
            dtype='<f4').reshape(shape).astype('float32')

        return wind_speed, terrain_slope, temperatures, velocities, firefronts

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns (T, C, H, W) float32 tensor for simulation idx.

        Validation:  deterministic, cached on first access.
        Training:    physical-param augmentation applied to raw fields before
                     feature engineering (so derived features stay consistent),
                     then tensor-space augmentation applied afterwards.
                     Cache stores the *clean* features; aug is applied fresh
                     each call, so each epoch sees a different realisation.
        """
        if self.Enable_validation:
            if idx not in self._cache:
                raw = self._load_raw(idx)
                self._cache[idx] = build_features(*raw)
            return torch.from_numpy(self._cache[idx])

        # Training path — cache clean features, augment on the fly
        if idx not in self._cache:
            raw = self._load_raw(idx)
            self._cache[idx] = (raw, build_features(*raw))

        raw_fields, clean_feats = self._cache[idx]

        # Physical augmentation: occasionally synthesise a new simulation variant
        if self.enable_aug and np.random.rand() < 0.6:
            feats = self.phys_aug(*raw_fields)
        else:
            feats = clean_feats.copy()

        x = torch.from_numpy(feats)   # (T, C, H, W)

        # Tensor-space augmentation (flips, roll, blur, noise)
        if self.enable_aug:
            x = self.aug_transform(x)

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window dataset  with activity-weighted oversampling
# ─────────────────────────────────────────────────────────────────────────────

def _window_activity(firefronts_seq: np.ndarray) -> float:
    """
    Measure how much fire transition happens in a frame window.
    Returns the number of newly ignited pixels across the sequence.
    firefronts_seq: (T, H, W)
    """
    diffs = np.diff(firefronts_seq.astype(np.float32), axis=0)
    return float(np.maximum(diffs, 0).sum())   # count of new ignitions


class FirePropagation(Dataset):
    """
    Sliding-window dataset over FireDataset simulations.

    Sampling strategy
    ─────────────────
    Uniform:             every valid window appears exactly once.
    Activity-weighted:   windows with active fire propagation (many newly
                         ignited pixels) appear more often, so the model
                         sees more examples of the interesting dynamics.
                         Implemented as weighted random sampling via the
                         `sample_weights` property — pass these to a
                         WeightedRandomSampler in the DataLoader.

    Oversampling fix
    ────────────────
    The original code had `for i in range(oversampling)` inside a loop
    `for i in range(len(...))`, causing the simulation index to be
    clobbered.  Renamed inner variable to `rep`.
    """

    def __init__(self, fire_dataset: FireDataset, config: dict):
        self.fire_dataset = fire_dataset
        self.in_frames    = config['in_frames']
        self.out_frames   = config['out_frames']
        total_frames      = self.in_frames + self.out_frames

        limit, oversampling = config.get('oversampling', [-1, 0])
        if fire_dataset.Enable_validation:
            limit, oversampling = -1, 0

        self.indices:  list[tuple[int, int]] = []
        self._weights: list[float]           = []

        for sim_idx in range(len(self.fire_dataset)):
            sim_data  = self.fire_dataset[sim_idx]           # (T, C, H, W)
            T         = sim_data.shape[0]
            fronts_np = sim_data[:, -1].numpy()              # (T, H, W) firefronts

            if T < total_frames:
                continue

            for t_start in range(T - total_frames + 1):
                t_mid = t_start + self.in_frames
                t_end = t_mid   + self.out_frames

                # Activity score for this window (used for weighted sampling)
                activity = _window_activity(fronts_np[t_start:t_end])
                n_copies = 1

                if oversampling > 0 and t_start > limit:
                    n_copies = oversampling   # fixed count for late-fire windows

                for rep in range(n_copies):   # ← was `for i`, which shadowed sim_idx
                    self.indices.append((sim_idx, t_start))
                    self._weights.append(max(activity, 1.0))   # at least weight 1

    # ── sampling weights for WeightedRandomSampler ────────────────────────────

    @property
    def sample_weights(self) -> torch.Tensor:
        """
        Use with torch.utils.data.WeightedRandomSampler to oversample
        windows with active fire propagation:

            from torch.utils.data import DataLoader, WeightedRandomSampler
            sampler = WeightedRandomSampler(ds.sample_weights, len(ds))
            loader  = DataLoader(ds, batch_size=8, sampler=sampler)
        """
        w = torch.tensor(self._weights, dtype=torch.float32)
        return w / w.sum()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sim_idx, t_start = self.indices[idx]
        full_data = self.fire_dataset[sim_idx]   # (T, C, H, W)

        t_mid = t_start + self.in_frames
        t_end = t_mid   + self.out_frames

        input_seq  = full_data[t_start:t_mid]          # (L_in, C, H, W)
        target_seq = full_data[t_mid:t_end, -1:]       # (L_out, 1, H, W)

        return input_seq, target_seq


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(data_cfg: dict, model_cfg: dict, batch_size: int = 4,
                     num_workers: int = 0, activity_weighted: bool = True):
    """
    Convenience function that returns (train_loader, val_loader).

    activity_weighted:  if True, use WeightedRandomSampler so windows with
                        active fire propagation are oversampled.
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_raw = FireDataset(data_cfg, Enable_validation=False)
    val_raw   = FireDataset(data_cfg, Enable_validation=True)

    train_ds = FirePropagation(train_raw, model_cfg)
    val_ds   = FirePropagation(val_raw,   model_cfg)

    if activity_weighted:
        sampler      = WeightedRandomSampler(train_ds.sample_weights, len(train_ds))
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, num_workers=num_workers,
                                  pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import tempfile, json

    # Minimal smoke test using synthetic data
    T, H, W = 30, 113, 32
    ff  = (np.random.rand(T, H, W) > 0.85).astype('float32')

    ws  = np.full((T, H, W), 5.0,  dtype='float32')
    ts  = np.full((T, H, W), 0.3,  dtype='float32')
    tmp = np.random.randn(T, H, W).astype('float32') * 100 + 600
    vel = np.random.randn(T, H, W).astype('float32') * 0.5 + 2

    feats = build_features(ws, ts, tmp, vel, ff)
    print(f"Feature tensor shape : {feats.shape}")      # (30, 16, 113, 32)
    assert feats.shape == (T, 16, H, W)

    aug = AugTransform()
    x   = torch.from_numpy(feats)
    xa  = aug(x)
    assert xa.shape == x.shape
    print(f"After AugTransform   : {xa.shape}")

    phys = PhysicalAug()
    fp   = phys(ws, ts, tmp, vel, ff)
    assert fp.shape == feats.shape
    print(f"After PhysicalAug    : {fp.shape}")
    print("All checks passed ✓")