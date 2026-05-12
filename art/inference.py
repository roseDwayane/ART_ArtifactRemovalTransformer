"""Offline inference pipeline for ART / IC-U-Net family.

The pipeline is:

    raw multichannel EEG csv
      -> read & channel-map to the model's 30-channel template
      -> resample to 256 Hz, 1-50 Hz FIR bandpass
      -> chunk into (30, 1024) windows
      -> run one of the four trained models
      -> stitch windows, resample back, write csv

The public entry point is :func:`reconstruct` (wraps :func:`decode_data`),
with :func:`preprocessing` / :func:`postprocessing` doing the I/O and channel
remapping. See ``main.py`` at the repository root for the canonical usage.
"""

import csv
import json
import os
import time

import numpy as np
import torch
from scipy.signal import decimate, firwin, lfilter, resample_poly

from .device import DEVICE
from .models import icunet, icunet_attn, icunet_pp, transformer
from .models import transformer_data

CHECKPOINT_ROOT = os.path.join('.', 'checkpoints')
SUPPORTED_MODELS = ('ICUNet', 'ICUNet++', 'ICUNet_attn', 'ART')


# ----------------------------------------------------------------------------
# Signal processing helpers
# ----------------------------------------------------------------------------

def resample(signal, fs, tgt_fs):
    """Resample multichannel signal to ``tgt_fs`` Hz."""
    if fs > tgt_fs:
        q = int(fs / tgt_fs)
        out = [decimate(ch, q) for ch in signal]
    elif fs < tgt_fs:
        p = int(tgt_fs / fs)
        out = [resample_poly(ch, p, 1) for ch in signal]
    else:
        out = signal
    return np.array(out).astype(np.float64)


def FIR_filter(signal, lowcut, highcut, fs=256.0, numtaps=1000):
    """Bandpass FIR filter (zero-pad delayed)."""
    fir_coeff = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    return lfilter(fir_coeff, 1.0, signal)


def read_csv_matrix(file_name):
    """Read a 2-D float CSV into a numpy array."""
    with open(file_name, 'r', newline='') as f:
        data = [row for row in csv.reader(f)]
    return np.array(data).astype(np.float64)


def save_csv_matrix(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv.writer(csvfile).writerows(data)


def cut_data(raw_data):
    """Split ``(C, T)`` into ``(C, 1024, T//1024)`` non-overlapping windows."""
    raw_data = np.array(raw_data).astype(np.float64)
    C, T = raw_data.shape
    num_windows = T // 1024
    total = np.zeros((C, 1024, num_windows))
    for i in range(num_windows):
        total[:, :, i] = raw_data[:, i * 1024:(i + 1) * 1024]
    return total


def glue_data(total):
    """Concatenate ``(C, 1024, N)`` windows back into ``(C, 1024*N)``.

    Adjacent windows are smoothed at the boundary by averaging the last sample
    of the previous window with the second sample of the next window.
    """
    glued = None
    for i in range(total.shape[2]):
        window = total[:, :, i]
        if i == 0:
            glued = window
        else:
            smooth = (glued[:, -1] + window[:, 1]) / 2
            glued[:, -1] = smooth
            window[:, 1] = smooth
            glued = np.append(glued, window, axis=1)
    return glued


# ----------------------------------------------------------------------------
# Channel mapping
# ----------------------------------------------------------------------------

def read_mapping_result(filename):
    with open(filename) as jsonfile:
        data = json.load(jsonfile)
    return data["mappingResult"], data["channelNum"], data["batch"]


def reorder_data(raw_data, mapping_result):
    """Re-map the user's channels to the model's 30-channel template."""
    new_data = np.zeros((30, raw_data.shape[1]))
    zero_arr = np.zeros((1, raw_data.shape[1]))

    for i, (indices, flag) in enumerate(
            zip(mapping_result["index"], mapping_result["isOriginalData"])):
        if flag is True:
            new_data[i, :] = raw_data[indices[0], :]
        elif indices[0] is None:
            new_data[i, :] = zero_arr
        else:
            stack = [raw_data[idx, :] for idx in indices]
            new_data[i, :] = np.mean(stack, axis=0)
    return new_data


def restore_order(data, all_data, mapping_result):
    """Inverse of :func:`reorder_data` — write model output back to the user's channel layout."""
    for i, (indices, flag) in enumerate(
            zip(mapping_result["index"], mapping_result["isOriginalData"])):
        if flag is True:
            all_data[indices[0], :] = data[i, :]
    return all_data


# ----------------------------------------------------------------------------
# Pre / post processing
# ----------------------------------------------------------------------------

def preprocessing(filename, samplerate, mapping_result):
    signal = read_csv_matrix(filename)
    signal = reorder_data(signal, mapping_result)
    signal = resample(signal, samplerate, 256)
    signal = FIR_filter(signal, 1, 50)
    return cut_data(signal)


def postprocessing(data, samplerate, outputfile, mapping_result, group_cnt, channel_num):
    data = resample(data, 256, samplerate)
    if group_cnt == 0:
        all_data = np.zeros((channel_num, data.shape[1]))
    else:
        all_data = read_csv_matrix(outputfile)
    all_data = restore_order(data, all_data, mapping_result)
    save_csv_matrix(all_data, outputfile)


# ----------------------------------------------------------------------------
# Model loading + forward
# ----------------------------------------------------------------------------

def _checkpoint_path(model_name):
    return os.path.join(CHECKPOINT_ROOT, model_name, 'modelsave', 'checkpoint.pth.tar')


def _load_checkpoint(resume_loc, model_name):
    if not os.path.isfile(resume_loc):
        raise FileNotFoundError(
            f"Checkpoint not found for model '{model_name}' at '{resume_loc}'.\n"
            f"Download the weights and place them at this path. "
            f"See README.md for the Google Drive link."
        )
    # weights_only=False keeps backward compatibility with checkpoints that
    # store optimizer / state metadata (default flipped to True in torch 2.6).
    return torch.load(resume_loc, map_location=DEVICE, weights_only=False)


def decode_data(data, std_num, mode):
    """Run a single ``(30, 1024)`` window through the requested model.

    Parameters
    ----------
    data : np.ndarray of shape ``(30, 1024)``
        Z-scored EEG window.
    std_num : float
        Original standard deviation (kept for API symmetry; not used by every model).
    mode : str
        One of ``SUPPORTED_MODELS``.
    """
    if mode == "ICUNet":
        model = icunet.UNet1(n_channels=30, n_classes=30).to(DEVICE)
        checkpoint = _load_checkpoint(_checkpoint_path(mode), mode)
        model.load_state_dict(checkpoint['state_dict'], False)
        model.eval()
        with torch.no_grad():
            x = torch.Tensor(data[np.newaxis, :, :]).to(DEVICE)
            decode = model(x)

    elif mode in ("ICUNet++", "ICUNet_attn"):
        if mode == "ICUNet++":
            model = icunet_pp.NestedUNet3(num_classes=30).to(DEVICE)
        else:
            model = icunet_attn.UNetpp3_Transformer(num_classes=30).to(DEVICE)
        checkpoint = _load_checkpoint(_checkpoint_path(mode), mode)
        model.load_state_dict(checkpoint['state_dict'], False)
        model.eval()
        with torch.no_grad():
            x = torch.Tensor(data[np.newaxis, :, :]).to(DEVICE)
            _, _, decode = model(x)

    elif mode == "ART":
        checkpoint = _load_checkpoint(_checkpoint_path(mode), mode)
        model = transformer.make_model(30, 30, N=2).to(DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(data).unsqueeze(0).to(DEVICE)
            batch = transformer_data.Batch(x, x, 0)
            out = model.forward(batch.src, batch.src[:, :, 1:], batch.src_mask, batch.trg_mask)
            decode = model.generator(out).permute(0, 2, 1)
            # The transformer drops one sample (sequence position) — pad it back so
            # the output shape matches the input window (30, 1024).
            pad = torch.zeros(1, 30, 1).to(DEVICE)
            decode = torch.cat((decode, pad), dim=2)

    else:
        raise ValueError(
            f"Unknown model name '{mode}'. Expected one of: {SUPPORTED_MODELS}."
        )

    return np.array(decode.cpu()).astype(np.float64)


def reconstruct(model_name, total, outputfile, group_cnt):
    """Decode every window in ``total`` and stitch them back together."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model name '{model_name}'. Expected one of: {SUPPORTED_MODELS}."
        )

    t0 = time.time()
    for i in range(total.shape[2]):
        window = np.squeeze(total[:, :, i])
        std = np.std(window)
        avg = np.average(window)
        window = (window - avg) / std
        total[:, :, i] = decode_data(window, std, model_name)[0]

    glued = glue_data(total)
    elapsed = time.time() - t0
    label = '{}-{}'.format(outputfile[:-4], group_cnt + 1)
    print(f"Using {model_name} model to reconstruct {label} has been success "
          f"in {elapsed:.3f} sec(s)")
    return glued
