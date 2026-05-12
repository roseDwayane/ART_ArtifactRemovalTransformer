"""Generate tiny fake EEG datasets for trainer smoke tests.

Creates:
    ./Real_EEG/{train,test,val}/{Brain,ChannelNoise,Eye,Heart,LineNoise,Muscle,Other}/seg_XXXX.csv
    ./MetaPreTrain/{Hyperscanning_navigation, Hyperscanning_slapjack, Lane_keeping}/3_ICA/sub_XXX_t_T.csv

Each csv is 30 channels x 1024 samples of float noise. The intent is *only* to
let the training pipelines run forward/backward/checkpoint end-to-end on a
machine without the real dataset.

Run from the project root:
    python training/make_fake_data.py
"""

import csv
import os
import numpy as np

CHANNELS = 30
SAMPLES = 1024
NUM_PER_SPLIT = 64  # how many segments per artifact class per split
NUM_PRETRAIN = 32   # how many MetaPreTrain segments per dataset

REAL_CLASSES = [
    "Brain", "ChannelNoise", "Eye", "Heart", "LineNoise", "Muscle", "Other",
]
SPLITS = ["train", "test", "val"]
PRETRAIN_DATASETS = [
    "Hyperscanning_navigation", "Hyperscanning_slapjack", "Lane_keeping",
]


def write_csv(path, arr):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(arr)


def gen_real_eeg(root="./Real_EEG", rng=None):
    rng = rng or np.random.default_rng(0)
    for split in SPLITS:
        for cls in REAL_CLASSES:
            d = os.path.join(root, split, cls)
            os.makedirs(d, exist_ok=True)
            for i in range(NUM_PER_SPLIT):
                arr = rng.standard_normal((CHANNELS, SAMPLES)).astype(np.float32)
                write_csv(os.path.join(d, f"seg_{i:04d}.csv"), arr)
        print(f"[fake-data] Real_EEG/{split}: "
              f"{NUM_PER_SPLIT} segments x {len(REAL_CLASSES)} classes")


def gen_metapretrain(root="./MetaPreTrain", rng=None):
    rng = rng or np.random.default_rng(1)
    for ds in PRETRAIN_DATASETS:
        d = os.path.join(root, ds, "3_ICA")
        os.makedirs(d, exist_ok=True)
        # the preDataset class expects filenames like "<prefix>_<t>.csv" and
        # tries to read t+4. We give each subject ids 0..N with 8 timesteps.
        for sub in range(NUM_PRETRAIN // 8):
            for t in range(8):
                arr = rng.standard_normal((CHANNELS, SAMPLES)).astype(np.float32)
                write_csv(os.path.join(d, f"sub_{sub:03d}_t_{t}.csv"), arr)
        print(f"[fake-data] MetaPreTrain/{ds}/3_ICA: "
              f"{NUM_PRETRAIN} segments")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    os.chdir(project_root)
    print(f"[fake-data] writing into {project_root}")
    gen_real_eeg()
    gen_metapretrain()
    print("[fake-data] done.")


if __name__ == "__main__":
    main()
