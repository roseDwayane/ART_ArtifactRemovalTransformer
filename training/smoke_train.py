"""Smoke-test every trainer.

Patches the hardcoded dataset sizes / epoch counts down to something tiny so
each pipeline (data loader + forward + backward + checkpoint save) runs in a
few seconds against the fake data created by ``make_fake_data.py``.

Run from the project root:

    python training/make_fake_data.py     # generate ./Real_EEG and ./MetaPreTrain
    python training/smoke_train.py        # run all four trainer smoke tests
"""

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)


def _patch_dataset_sizes():
    """Shrink hardcoded dataset lengths so a smoke test finishes quickly."""
    import numpy as np
    import torch
    from training import datasets_art as da
    from training import datasets_icunet as di

    TINY = 8
    for cls in (di.myDataset, da.myDataset, da.preDataset):
        cls.lenth = TINY
        cls.lenthtest = TINY
        cls.lenthval = TINY
        cls.__len__ = lambda self, _t=TINY: _t

    # preDataset has hardcoded dataset-loc boundaries (249816 / 506365). Replace
    # __getitem__ with one that reads the first 8 files of Hyperscanning_navigation.
    def _pretrain_getitem(self, idx):
        folder = './MetaPreTrain/Hyperscanning_navigation/3_ICA/'
        files = sorted(f for f in os.listdir(folder) if f.endswith('.csv'))
        arr = np.loadtxt(folder + files[idx % len(files)], delimiter=',',
                         dtype=np.float64)
        attr = (arr - arr.mean()) / (arr.std() + 1e-10)
        ys = torch.ones(30, 1023)
        return (torch.tensor(attr, dtype=torch.float32),
                torch.tensor(attr.copy(), dtype=torch.float32),
                ys)

    da.preDataset.__getitem__ = _pretrain_getitem


def _shrink_args(args):
    args.max_epochs = 1
    args.num_workers = 0
    args.batch_size = 2
    args.milestones = [10]
    return args


def run_tf_train():
    print("\n========== train_art_simple.py ==========")
    _patch_dataset_sizes()
    import torch
    from art.device import DEVICE
    from art.models.transformer import make_model
    from training import train_art_simple
    from training.datasets_art import data_load, myDataset
    from training.losses_art import LabelSmoothing, SimpleLossCompute
    from training.optim import NoamOpt

    V = 30
    model = make_model(V, V, N=2).to(DEVICE)
    crit = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0).to(DEVICE)
    opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                  torch.optim.Adam(model.parameters(), lr=0,
                                   betas=(0.9, 0.98), eps=1e-9))
    trainset = myDataset(mode=0, train_len=8, block_num=True)
    loader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True,
                                         num_workers=0)
    model.train()
    train_art_simple.run_epoch(data_load(loader, device=DEVICE), model,
                               SimpleLossCompute(model.generator, crit, opt))
    print("train_art_simple.py OK")


def run_tf_train2():
    print("\n========== train_art.py ==========")
    _patch_dataset_sizes()
    import torch
    from art.device import DEVICE
    from art.models.transformer import make_model
    from training import train_art
    from training.datasets_art import data_load, myDataset
    from training.losses_art import LabelSmoothing, SimpleLossCompute
    from training.optim import NoamOpt
    from training.plot_helpers import save_checkpoint

    args = train_art.model_train_parameter(0, './smoke_train_art',
                                           './smoke_data', train_len=8)
    args = _shrink_args(args)
    args.resume = False

    V = 30
    model = make_model(V, V, N=2).to(DEVICE)
    crit = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0).to(DEVICE)
    opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                  torch.optim.Adam(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.98), eps=1e-9))
    trainset = myDataset(mode=0, train_len=8, block_num=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=0)
    valset = myDataset(mode=2, block_num=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0)

    os.makedirs(args.savedir, exist_ok=True)
    model.train()
    train_art.run_epoch(data_load(train_loader, device=DEVICE), model,
                        SimpleLossCompute(model.generator, crit, opt))
    model.eval()
    train_art.run_epoch(data_load(val_loader, device=DEVICE), model,
                        SimpleLossCompute(model.generator, crit, opt))
    save_checkpoint({'epoch': 1, 'state_dict': model.state_dict()}, args.savedir)
    print("train_art.py OK; checkpoint at", args.savedir)


def run_tf_pre_train():
    print("\n========== pretrain_art.py ==========")
    _patch_dataset_sizes()
    import torch
    from art.device import DEVICE
    from art.models.transformer import make_model
    from training import pretrain_art
    from training.datasets_art import data_load, myDataset, preDataset
    from training.losses_art import LabelSmoothing, SimpleLossCompute
    from training.optim import NoamOpt
    from training.plot_helpers import save_checkpoint

    args = pretrain_art.model_train_parameter(0, './smoke_pretrain_art',
                                              './smoke_data', train_len=8)
    args = _shrink_args(args)
    args.resume = False

    V = 30
    model = make_model(V, V, N=2).to(DEVICE)
    crit = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0).to(DEVICE)
    opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                  torch.optim.Adam(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.98), eps=1e-9))
    trainset = preDataset(mode=0, train_len=8, block_num=True)
    loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=0)
    valset = myDataset(mode=2, block_num=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0)
    os.makedirs(args.savedir, exist_ok=True)
    model.train()
    pretrain_art.run_epoch(data_load(loader, device=DEVICE), model,
                           SimpleLossCompute(model.generator, crit, opt))
    model.eval()
    pretrain_art.run_epoch(data_load(val_loader, device=DEVICE), model,
                           SimpleLossCompute(model.generator, crit, opt))
    save_checkpoint({'epoch': 1, 'subfold': 1,
                     'state_dict': model.state_dict()}, args.savedir)
    print("pretrain_art.py OK; checkpoint at", args.savedir)


def run_train():
    print("\n========== train_icunet.py (IC-U-Net family) ==========")
    _patch_dataset_sizes()
    import torch
    import torch.nn as nn
    from art.device import DEVICE
    from art.models import icunet_pp
    from training import train_icunet
    from training.datasets_icunet import myDataset as DS

    args = train_icunet.model_train_parameter(loss=[1, 1, 1, 1],
                                              save='./smoke_train_icunet',
                                              data='./smoke_data', train_len=8,
                                              arch_num=3, block=1)
    args = _shrink_args(args)
    args.resume = False
    args.savedir = args.save + '/modelsave'
    args.savefig = args.save + '/result'

    model = icunet_pp.NestedUNet3(num_classes=30).to(DEVICE)
    crit = nn.MSELoss().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)

    trainset = DS(mode=0, iter=30, data=args.savedata, train_len=8,
                  block_num=args.block_num)
    loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=0)
    os.makedirs(args.savedir, exist_ok=True)
    train_icunet.train(args, loader, model, crit, opt, epoch=0)
    train_icunet.save_checkpoint(
        {'epoch': 1, 'state_dict': model.state_dict(),
         'epochs_since_improvement': 0, 'best_loss': 1.0,
         'lossTr': 0, 'lossTs': 0, 'lossVal': 0, ' lr': args.lr},
        False, args.savedir,
    )
    print("train_icunet.py OK; checkpoint at", args.savedir)


def main():
    print("Smoke-testing trainers (using fake data in ./Real_EEG and ./MetaPreTrain).")
    print("Requires you to have run `python training/make_fake_data.py` first.\n")

    tests = [
        ("train_art_simple", run_tf_train),
        ("train_art",        run_tf_train2),
        ("pretrain_art",     run_tf_pre_train),
        ("train_icunet",     run_train),
    ]
    results = {}
    for name, fn in tests:
        try:
            fn()
            results[name] = "OK"
        except Exception as e:
            traceback.print_exc()
            results[name] = f"FAIL: {type(e).__name__}: {e}"

    print("\n\n========== summary ==========")
    for name, status in results.items():
        print(f"  {name:25s} {status}")
    failed = [n for n, s in results.items() if not s.startswith("OK")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
