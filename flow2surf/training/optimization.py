"""Learning-rate schedule, gradient clipping, and parameter EMA."""

import copy
import math

import jittor as jt


class ModelEMA:
    """Maintain a frozen evaluation model with parameter-wise EMA updates."""

    def __init__(self, model, decay):
        decay = float(decay)
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for param in self.model.parameters():
            param.stop_grad()
        current = model.state_dict()
        average = self.model.state_dict()
        if current.keys() != average.keys():
            raise ValueError("EMA model does not match source model")
        self._state_pairs = tuple((current[name], average[name]) for name in current)
        jt.sync_all()

    def update(self):
        """Update the moving average after one optimizer step."""
        with jt.no_grad():
            for current, average in self._state_pairs:
                value = current.detach()
                average.update(self.decay * average + (1.0 - self.decay) * value)
        jt.sync_all(False)


def clip_grad_norm(optimizer, max_norm):
    """Clip the global trainable-gradient L2 norm and return it on-device."""
    squared_norms = [
        (gradient * gradient).sum()
        for group in optimizer.param_groups
        for param, gradient in zip(group["params"], group["grads"])
        if not param.is_stop_grad()
    ]
    total_norm = jt.sqrt(sum(squared_norms)) if squared_norms else jt.float32(0.0)
    coefficient = jt.minimum(float(max_norm) / (total_norm + 1e-6), 1.0)
    for group in optimizer.param_groups:
        for param, gradient in zip(group["params"], group["grads"]):
            if not param.is_stop_grad():
                gradient.update(gradient * coefficient)
    return total_norm.stop_grad()


def get_lr(cfg, epoch):
    """Return the warmup and cosine-decay learning rate for an epoch."""
    lr_max = cfg["lr"]
    lr_min = cfg["lr_min"]
    warmup = cfg["warmup_epochs"]
    duration = cfg["pretrain_epochs"]
    if epoch > duration:
        progress = (epoch - duration - 1) / max(cfg["epochs"] - duration - 1, 1)
        return lr_min + 0.5 * (cfg["refine_lr"] - lr_min) * (1 + math.cos(math.pi * progress))
    if warmup > 0 and epoch <= warmup:
        return lr_max * epoch / warmup
    progress = (epoch - warmup - 1) / max(duration - warmup - 1, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
