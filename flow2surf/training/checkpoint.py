"""Resumable training checkpoints and readable history."""

import json
import logging
import os
from pathlib import Path
import tempfile

import jittor as jt
import numpy as np

log = logging.getLogger(__name__)


def _numpy_state(value):
    """Copy nested optimizer/model state without live Jittor variables."""
    if isinstance(value, jt.Var):
        return value.numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _numpy_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_numpy_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_numpy_state(item) for item in value)
    return value


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.stem + "_", suffix=path.suffix, dir=path.parent
    )
    os.close(fd)
    try:
        writer(temporary)
        with open(temporary, "rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _restore_weights(model, state):
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("Checkpoint parameter names do not match the model")
    for name, value in state.items():
        if tuple(value.shape) != tuple(expected[name].shape):
            raise ValueError(f"Checkpoint shape mismatch: {name}")
    model.load_parameters(state)


def load_model_weights(model, path):
    """Load checkpoint EMA/raw weights or a model-only weight export."""
    state = jt.load(str(path))
    model_config = None
    if "optimizer" in state and "model" in state:
        model_config = state["state"]["config"]
        weights = state["ema"] if state["ema"] is not None else state["model"]
    elif "model" in state:
        model_config = state.get("model_config")
        weights = state["model"]
    else:
        weights = state
    # Graph choice changes computation without changing parameter shapes.
    if model_config is not None and "encoder_graph" in model_config:
        if model_config["encoder_graph"] != "xyz":
            raise ValueError("Checkpoint does not use the maintained XYZ encoder")
    _restore_weights(model, weights)


def read_resume(path):
    """Accept only the current full-resume format; exports are not resumes."""
    payload = jt.load(str(path))
    required = {"model", "optimizer", "ema", "state"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(
            "--resume requires a full training checkpoint, not model-only weights"
        )
    if set(payload["state"]) != {
        "epoch",
        "checkpoint",
        "checkpoint_epoch",
        "history",
        "config",
        "dataset",
    }:
        raise ValueError(
            "Resume checkpoint does not contain the current training state schema"
        )
    return payload


class CheckpointManager:
    """Save resumable checkpoints, final inference weights, and progress."""

    def __init__(self, output_dir):
        self.state_path = Path(output_dir) / "train_state.json"
        self.dir = Path(output_dir) / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)

    def write_state(self, state):
        def write(temporary):
            with open(temporary, "w") as file:
                json.dump(state, file, indent=2, allow_nan=False)
                file.write("\n")

        _atomic_write(self.state_path, write)

    def save_checkpoint(self, model, optimizer, ema, state):
        path = self.dir / f"epoch_{state['epoch']:03d}.pkl"
        defaults = optimizer.state_dict()["defaults"]
        # Parameters are already in model; preserve optimizer moments/counters.
        defaults = {key: value for key, value in defaults.items() if key != "params"}
        defaults["param_groups"] = [
            {key: value for key, value in group.items() if key != "params"}
            for group in defaults["param_groups"]
        ]
        optimizer_state = _numpy_state({"defaults": defaults})
        state["checkpoint"] = path.name
        state["checkpoint_epoch"] = state["epoch"]
        payload = {
            "model": _numpy_state(model.state_dict()),
            "optimizer": optimizer_state,
            "ema": _numpy_state(ema.model.state_dict()) if ema is not None else None,
            "state": state,
        }
        _atomic_write(path, lambda temporary: jt.save(payload, temporary))
        log.info("Checkpoint: %s", path)
        return str(path)

    def save_final(self, model):
        """Save only the final inference model's parameter dictionary."""
        path = self.dir / "final.pkl"
        weights = _numpy_state(model.state_dict())
        _atomic_write(path, lambda temporary: jt.save(weights, temporary))
        log.info("Final model: %s", path)
        return str(path)

    def restore(self, model, optimizer, ema, payload):
        if (ema is None) != (payload["ema"] is None):
            raise ValueError("Checkpoint EMA setting does not match the config")
        _restore_weights(model, payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if ema is not None:
            _restore_weights(ema.model, payload["ema"])
        state = payload["state"]
        log.info("Restored model, optimizer, EMA and training state")
        return state
