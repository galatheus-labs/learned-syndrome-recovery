from dataclasses import dataclass

import yaml

from parity import ConstraintConfig


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 8
    lr: float = 1e-4
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    num_layers: int = 4
    base_channels: int = 32
    parity_weight: float = 1.0
    dice_weight: float = 2.0
    self_consistent_prob: float = 0.0
    direct_decoder: bool = False
    seed_decoder: bool = False
    refiner_steps: int = 0
    sideband_bits: int = 0
    coarse_map: bool = False
    coarse_size: int = 4
    coarse_threshold: float = 0.5
    dct_field: bool = False
    dct_keep_freqs: int = 4
    adaptive_dct: bool = False
    adaptive_dct_max_freqs: int = 6
    rate_weight: float = 0.001
    nullspace_aug_prob: float = 0.5
    embed_weight: float = 0.0
    resnet: bool = False
    resnet_width: int = 128
    resnet_depth: int = 20
    boltzmann_anneal: bool = False
    boltzmann_J: float = 0.5
    boltzmann_sweeps: int = 5
    bp_decoder: bool = False
    bp_polish: bool = False
    bp_hidden_dim: int = 64
    bp_iters: int = 10
    val_every: int = 0
    # Ablation switches — disable core components for controlled experiments
    disable_projection: bool = False      # skip parity projection (project_with_left_inverse)
    disable_syndrome_cond: bool = False   # zero out syndrome conditioning to model
    disable_x_base: bool = False          # replace canonical base with zeros
    random_constraints: bool = False      # use random H matrix with matched rank instead of structured parity
    bp_prior: bool = False                # replace x_base with BP marginals as input channel
    bp_prior_iters: int = 20             # BP iterations for marginal computation
    bp_prior_damping: float = 0.3        # BP damping factor

@dataclass
class DiffusionConfig:
    T: int = 1000
    schedule: str = "cosine"

@dataclass
class DataConfig:
    path: str = ""
    val_path: str = ""
    num_workers: int = 4

@dataclass
class LoggingConfig:
    project: str = "parity_diffusion"
    run_name: str = "run"
    ckpt_every: int = 1000

class Config:
    def __init__(self):
        self.seed: int = 0
        self.cube_size: int = 32
        self.train: TrainConfig = TrainConfig()
        self.diffusion: DiffusionConfig = DiffusionConfig()
        self.data: DataConfig = DataConfig()
        self.logging: LoggingConfig = LoggingConfig()
        self.constraints: ConstraintConfig = ConstraintConfig()


def load_config(path: str) -> Config:
    """Parse a YAML config file into a Config object."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    cfg = Config()
    for k, v in data.items():
        if k == "constraints":
            cfg.constraints = ConstraintConfig(**v)
        elif isinstance(v, dict) and hasattr(cfg, k):
            sub = getattr(cfg, k)
            for kk, vv in v.items():
                setattr(sub, kk, vv)
        else:
            setattr(cfg, k, v)
    return cfg
