"""Inference-time helpers for the ART transformer.

Only :class:`Batch` is needed by the offline pipeline. Training-time dataset
classes live in ``training/datasets_art.py``.
"""

import torch
from torch.autograd import Variable

from .transformer import subsequent_mask


class Batch:
    """Container for a batch of src/trg tensors with masks."""

    def __init__(self, src, trg=None, ys=None, pad=0):
        self.src = src
        self.ys = ys
        self.src_len = src[:, 0, :]
        self.src_mask = (self.src_len != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :, :-1]
            self.trg_len = trg[:, 0, :]
            self.trg_x = self.trg_len[:, :-1]
            self.trg_y = self.trg_len[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg_x, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Mask to hide padding and future positions."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
