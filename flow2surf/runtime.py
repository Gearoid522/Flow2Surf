"""Logging, reproducibility, configuration, and timestamps."""

import logging
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

import jittor as jt
import numpy as np


INFERENCE_FIELDS = (
    "num_steps", "alpha_clean_power", "alpha_noisy_power", "seed_k",
    "patch_agg_beta", "inference_chunks", "inference_seed",
)


def configure_logging(log_file=None):
    """Configure the Flow2Surf logger for one command-line process."""
    logger = logging.getLogger("flow2surf")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def seed_runtime(seed):
    """Seed Python, NumPy, and Jittor with one integer seed."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def git_revision(repo=None):
    """Return the short Git revision, marked dirty when the worktree differs."""
    repo = repo or os.getcwd()
    try:
        commit = (
            subprocess.check_output(
                ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        status = subprocess.check_output(
            ["git", "-C", repo, "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        )
        return f"{commit}{'-dirty' if status.strip() else ''}"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def timestamp():
    """Return a sortable local timestamp for output filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def format_duration(seconds):
    """Format elapsed time or ETA as compact hours, minutes, and seconds."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s"


def load_config(path):
    """Require exactly the maintained configuration fields, without aliases."""
    with open(path) as file:
        cfg = yaml.safe_load(file)
    default_path = Path(__file__).resolve().parent.parent / "configs/default.yaml"
    with default_path.open() as file:
        schema = yaml.safe_load(file)
    if not isinstance(cfg, dict) or set(cfg) != set(schema):
        keys = set(cfg) if isinstance(cfg, dict) else set()
        raise ValueError(
            f"Invalid config schema: missing={sorted(set(schema)-keys)}, "
            f"extra={sorted(keys-set(schema))}"
        )
    return cfg
