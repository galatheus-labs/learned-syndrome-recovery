from contextlib import contextmanager
from tqdm import tqdm
# import wandb


# def init_wandb(cfg):
#     wandb.init(project=cfg.logging.project, name=cfg.logging.run_name, config=cfg)


# def log_metrics(step, **metrics):
#     wandb.log({k: v for k, v in metrics.items()}, step=step)


@contextmanager
def tqdm_bar(*args, **kwargs):
    bar = tqdm(*args, **kwargs)
    try:
        yield bar
    finally:
        bar.close()
