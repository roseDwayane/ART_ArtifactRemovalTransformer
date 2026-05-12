"""Shared device helper for the ART training scripts.

The original code hard-coded ``.cuda()`` everywhere. This helper picks the best
available device (CUDA -> MPS -> CPU) so the same scripts also run on macOS
(Apple Silicon) and on CPU-only machines.
"""

import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = get_device()
ON_GPU = DEVICE.type == 'cuda'
