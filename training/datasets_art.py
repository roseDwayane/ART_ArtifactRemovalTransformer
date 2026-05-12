"""Training-time datasets and data-loading utilities for the ART transformer.

* :class:`myDataset` reads paired clean / noisy EEG segments from
  ``Real_EEG/{train,test,val}/<artifact_class>/*.csv``.
* :class:`preDataset` reads pretraining segments from
  ``MetaPreTrain/<dataset>/3_ICA/*.csv``, pairing each ``t`` second clip with
  the ``t + 4`` clip of the same recording.
* :func:`data_load` wraps a DataLoader so each batch is moved to ``DEVICE``
  and packaged into a :class:`Batch`.
"""

import csv
import os
import random

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from art.device import DEVICE
from art.models.transformer_data import Batch


# ----------------------------------------------------------------------------
# Toy generator (kept for parity with the original training scripts)
# ----------------------------------------------------------------------------

def data_gen(V, batch, nbatches):
    """Yield random `Batch` objects for sanity checking the optimization loop."""
    for _ in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


def data_load(train_loader, device=None):
    """Iterate over a DataLoader, moving each batch onto ``device``."""
    if device is None:
        device = DEVICE
    print("data_load:", len(train_loader))
    for attr, target, ys in train_loader:
        src = attr.to(device)
        tgt = target.to(device)
        ys = ys.to(device)
        yield Batch(src, tgt, ys, 0)


# ----------------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------------

def _read_csv(file_name):
    with open(file_name, 'r', newline='') as f:
        return np.array([row for row in csv.reader(f)]).astype(np.float64)


# ----------------------------------------------------------------------------
# Real_EEG dataset (train / test / val)
# ----------------------------------------------------------------------------

ARTIFACT_CLASSES = [
    "Brain", "ChannelNoise", "Eye", "Heart", "LineNoise", "Muscle", "Other",
]


class myDataset(Dataset):
    """Loads paired (clean, noisy) EEG segments from ``Real_EEG/<split>/``.

    Each item returns ``(noisy, clean, ys)`` where ``ys`` is a constant
    sequence used as decoder input by the transformer.
    """

    def __init__(self, mode, train_len=0, block_num=1):
        self.sample_rate = 256
        self.lenth = train_len
        self.lenthtest = 3600
        self.lenthval = 3500
        self.mode = mode  # 0=train, 1=test, 2=val
        self.block_num = block_num

    def __len__(self):
        if self.mode == 2:
            return self.lenthval
        if self.mode == 1:
            return self.lenthtest
        return self.lenth

    def _split_root(self):
        return {0: 'train', 1: 'test', 2: 'val'}[self.mode]

    def __getitem__(self, idx):
        split_root = f'./Real_EEG/{self._split_root()}'
        brain_dir = f'{split_root}/Brain/'

        all_files = os.listdir(brain_dir)
        file_name = brain_dir + all_files[idx]
        data_clean = _read_csv(file_name)

        # try up to 7 random artifact classes to find a matching noisy version
        data_noise = data_clean
        for _ in range(7):
            cls = ARTIFACT_CLASSES[random.randint(0, 6)]
            candidate = f'{split_root}/{cls}/' + all_files[idx]
            if os.path.isfile(candidate):
                data_noise = _read_csv(candidate)
                break

        data_avg = np.average(data_noise)
        data_std = np.std(data_noise)
        if int(data_std) != 0:
            target = ((data_clean - data_avg) / data_std).astype(np.float64)
            attr = ((data_noise - data_avg) / data_std).astype(np.float64)
        else:
            target = (data_clean - data_avg).astype(np.float64)
            attr = (data_noise - data_avg).astype(np.float64)

        target = torch.FloatTensor(target)
        attr = torch.FloatTensor(attr)
        ys = torch.ones(30, 1023).fill_(1).type_as(attr)
        return attr, target, ys


# ----------------------------------------------------------------------------
# MetaPreTrain dataset (pretraining only)
# ----------------------------------------------------------------------------

PRETRAIN_DATASETS = ["Hyperscanning_navigation", "Hyperscanning_slapjack", "Lane_keeping"]
# These boundaries match the original training corpus. If you train on
# a different corpus, adjust them (or subclass preDataset).
PRETRAIN_DATALOC_BOUNDARIES = [(0, 249816, 0), (249816, 506365, 1), (506365, 870300, 2)]


class preDataset(Dataset):
    """Self-supervised pretraining dataset.

    Given segment ``X_t.csv`` we try to load ``X_{t+4}.csv`` as the target —
    i.e. the model is asked to predict 4 seconds into the future.
    """

    def __init__(self, mode, train_len=870300, block_num=1):
        self.sample_rate = 256
        self.lenth = train_len
        self.lenthtest = 3600
        self.lenthval = 3500
        self.mode = mode
        self.block_num = block_num

    def __len__(self):
        if self.mode == 2:
            return self.lenthval
        if self.mode == 1:
            return self.lenthtest
        return self.lenth

    def __getitem__(self, idx):
        for lo, hi, dataloc in PRETRAIN_DATALOC_BOUNDARIES:
            if lo <= idx < hi:
                now_idx = idx - lo
                break
        else:
            raise IndexError(f"idx {idx} out of range")

        folder = f'./MetaPreTrain/{PRETRAIN_DATASETS[dataloc]}/3_ICA/'
        all_files = os.listdir(folder)
        file_name1 = folder + all_files[now_idx]

        # build the "+4 seconds" filename
        parts = all_files[now_idx].split('_')
        numeric_part = parts[-1].split('.')[0]
        parts[-1] = f'{int(numeric_part) + 4}.csv'
        file_name2 = folder + '_'.join(parts)

        try:
            data_noise = _read_csv(file_name1)
            data_clean = _read_csv(file_name2)
        except Exception:
            data_noise = _read_csv(file_name1)
            data_clean = data_noise

        data_avg = np.average(data_noise)
        data_std = np.std(data_noise)
        if int(data_std) != 0:
            target = ((data_clean - data_avg) / data_std).astype(np.float64)
            attr = ((data_noise - data_avg) / data_std).astype(np.float64)
        else:
            target = (data_clean - data_avg).astype(np.float64)
            attr = (data_noise - data_avg).astype(np.float64)

        target = torch.FloatTensor(target)
        attr = torch.FloatTensor(attr)
        ys = torch.ones(30, 1023).fill_(1).type_as(attr)
        return attr, target, ys
