import time
import torch
from tf_opt import NoamOpt
from tf_loss import LabelSmoothing, SimpleLossCompute
from tf_model import make_model
from tf_data import data_gen, data_load, myDataset


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        #print("run_epoch1:", batch.src.shape)
        #print("run_epoch2:", batch.trg.shape)
        #print("run_epoch1:", batch.src_mask.shape)
        #print("run_epoch2:", batch.trg_mask.shape)
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        #loss = loss_compute(out, batch.trg_y, batch.ntokens)
        loss = loss_compute(out, batch.trg, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        #if i % 50 == 1:
        elapsed = time.time() - start
        print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
        start = time.time()
        tokens = 0
    return total_loss / total_tokens


V = 30
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
trainset = myDataset(mode=0, train_len=39400, block_num=True)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)

for epoch in range(10):
    model.train()
    #run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
    run_epoch(data_load(train_loader, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model,
                    SimpleLossCompute(model.generator, criterion, None)))