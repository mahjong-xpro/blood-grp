"""Blood-v2: Superhuman Bloody Battle Mahjong AI."""

# PyTorch 2.6 changed torch.load default to weights_only=True, but Sample Factory
# checkpoints contain numpy scalars (numpy.core.multiarray.scalar) that fail safe
# unpickling.  Patch torch.load at import time so it defaults to weights_only=False.
# This takes effect in both the main process and SF2 worker subprocesses (which
# import blood.model.factory → blood → this __init__.py before loading checkpoints).
import torch as _torch

_original_torch_load = _torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


_torch.load = _patched_torch_load
