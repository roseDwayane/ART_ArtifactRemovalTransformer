import time
import torch
from art.device import DEVICE
from training.optim import NoamOpt
from training.losses_art import LabelSmoothing, SimpleLossCompute
from art.models.transformer import make_model
from training.datasets_art import data_gen, data_load, myDataset


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        elapsed = time.time() - start
        print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
        start = time.time()
        tokens = 0
    return total_loss / total_tokens


def main():
    print(f"[INFO] training device = {DEVICE}")

    V = 30
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0).to(DEVICE)
    model = make_model(V, V, N=2).to(DEVICE)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    trainset = myDataset(mode=0, train_len=39400, block_num=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)

    for epoch in range(10):
        model.train()
        run_epoch(data_load(train_loader, device=DEVICE), model, SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model,
                        SimpleLossCompute(model.generator, criterion, None)))


if __name__ == '__main__':
    main()
