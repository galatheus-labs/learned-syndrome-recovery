#!/usr/bin/env python3
"""Generate ablation config YAML files from the base config."""
import yaml
from pathlib import Path

base_path = Path(__file__).parent / "ablation_base.yaml"
with open(base_path) as f:
    base = yaml.safe_load(f)


def write_variant(name: str, overrides: dict):
    """Create a variant config by deep-merging overrides into the base."""
    import copy
    cfg = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    cfg["logging"]["run_name"] = f"ablation_{name}"
    out = Path(__file__).parent / f"ablation_{name}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  {out.name}")


print("Generating ablation configs:")

# --- Core ablations ---

# 1. Full model (already the base, but write explicitly for completeness)
write_variant("full", {})

# 2. No coarse map
write_variant("no_coarse", {"train": {"coarse_map": False}})

# 3. No spectral field (adaptive DCT)
write_variant("no_spectral", {"train": {"adaptive_dct": False}})

# 4. No BP loss (parity_weight = 0)
write_variant("no_bp_loss", {"train": {"parity_weight": 0.0}})

# 5. No null-space augmentation
write_variant("no_nullaug", {"train": {"nullspace_aug_prob": 0.0}})

# 6. Algebraic only (syndrome + x_base + projection, no coarse or spectral)
write_variant("algebraic_only", {"train": {"coarse_map": False, "adaptive_dct": False}})

# 7. Side-information only (no x_base, no syndrome cond, no projection,
#    no parity loss, no null-space aug — pure spatial decoder)
write_variant("sideinfo_only", {
    "train": {
        "disable_x_base": True,
        "disable_syndrome_cond": True,
        "disable_projection": True,
        "parity_weight": 0.0,
        "nullspace_aug_prob": 0.0,
    }
})

# 8. No projection (keep training, skip project_with_left_inverse)
write_variant("no_projection", {"train": {"disable_projection": True}})

# 9. No syndrome conditioning (keep x_base and projection)
write_variant("no_syndrome_cond", {"train": {"disable_syndrome_cond": True}})

# 10. No x_base (keep syndrome conditioning and projection)
write_variant("no_xbase", {"train": {"disable_x_base": True}})

# --- Structural controls ---

# 11. Axis + global only (fewer constraints than full axis+plane+global)
write_variant("axis_global_only", {
    "constraints": {
        "axis_parity": True,
        "plane_parity": False,
        "global_parity": True,
    }
})

# 12. Random constraints (matched rank)
write_variant("random_constraints", {"train": {"random_constraints": True}})

# --- Width sweep ---

# 13. Width 128
write_variant("w128", {"train": {"resnet_width": 128}})

# 14. Width 256
write_variant("w256", {"train": {"resnet_width": 256}})

print("\nDone. 14 ablation configs generated.")
