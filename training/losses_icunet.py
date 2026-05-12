import torch
import torch.nn as nn
import numpy as np
import time


def lossFunc(args, output, target):
    criterion = nn.MSELoss()
    loss1 = criterion(output, target)

    doutput = output[:, :, 1:] - output[:, :, :-1]
    dtarget = target[:, :, 1:] - target[:, :, :-1]
    loss2 = criterion(doutput, dtarget)

    d2output = doutput[:, :, 1:] - doutput[:, :, :-1]
    d2target = dtarget[:, :, 1:] - dtarget[:, :, :-1]
    loss3 = criterion(d2output, d2target)

    # freq_MSE
    output_freq = torch.fft.rfft(output, 1)
    output_freq = torch.stack((output_freq.real, output_freq.imag), -1)
    target_freq = torch.fft.rfft(target, 1)
    target_freq = torch.stack((target_freq.real, target_freq.imag), -1)

    output_freq = sum(abs(output_freq.T)) / len(output_freq.T)
    target_freq = sum(abs(target_freq.T)) / len(target_freq.T)

    output_freq = output_freq / torch.std(target_freq)
    target_freq = target_freq / torch.std(target_freq)

    lossf = criterion(output_freq, target_freq)

    loss = args.loss[0] * loss1 + args.loss[1] * loss2 + args.loss[2] * loss3 + args.loss[3] * lossf


    return loss1, loss2, loss3, lossf, loss

def lossArch3(args, output1, output2, output3, target, i, total_batches, start_time):

    #output1, output2, output3 = model(input)  # output = [output1, output2, output3, output4]

    loss1, loss2, loss3, lossf, loss0_1 = lossFunc(args, output1, target)
    lossTotal = [loss1.item(), loss2.item(), loss3.item(), lossf.item(), loss0_1.item()]

    loss1, loss2, loss3, lossf, loss0_2 = lossFunc(args, output2, target)
    lossTotal = np.append(lossTotal, [loss1.item(), loss2.item(), loss3.item(), lossf.item(), loss0_2.item()])

    loss1, loss2, loss3, lossf, loss0_3 = lossFunc(args, output3, target)
    lossTotal = np.append(lossTotal, [loss1.item(), loss2.item(), loss3.item(), lossf.item(), loss0_3.item()])

    total_loss = loss0_1 + loss0_2 + loss0_3

    #epoch_loss.append(total_loss)
    time_taken = time.time() - start_time

    print('[%d/%d] loss1-0_1: %.6f loss2-0_1: %.6f loss3-0_1: %.6f lossf-0_1: %.6f loss0_1: %.6f\n'
          'loss1-0_2: %.6f loss2-0_2: %.6f loss3-0_2: %.6f lossf-0_2: %.6f loss0_2: %.6f\n'
          'loss1-0_3: %.6f loss2-0_3: %.6f loss3-0_3: %.6f lossf-0_3: %.6f loss0_3: %.6f\n'
          'total_loss: %.6f time:%.2f' % (
              i, total_batches, lossTotal[0], lossTotal[1], lossTotal[2], lossTotal[3], lossTotal[4],
              lossTotal[5], lossTotal[6], lossTotal[7], lossTotal[8], lossTotal[9],
              lossTotal[10], lossTotal[11], lossTotal[12], lossTotal[13], lossTotal[14],
              total_loss, time_taken))
    return lossTotal, total_loss

def lossArch4(args, output1, output2, output3, output4, target, i, total_batches, start_time):

    #output1, output2, output3, output4= model(input)  # output = [output1, output2, output3, output4]

    loss1, loss2, loss3, lossf, loss0_1 = lossFunc(args, output1, target)
    lossTotal = [loss1.item(), loss2.item(), loss3.item(), lossf.item(), loss0_1.item()]

    loss1, loss2, loss3, lossf, loss0_2 = lossFunc(args, output2, target)
    lossTotal = np.append(lossTotal, [loss1.item(), loss2.item(), loss3.item(), lossf.item(), loss0_2.item()])

    loss1, loss2, loss3, lossf, loss0_3 = lossFunc(args, output3, target)
    lossTotal = np.append(lossTotal, [loss1.item(), loss2.item(), loss3.item(), lossf.item(), loss0_3.item()])

    loss1, loss2, loss3, lossf, loss0_4 = lossFunc(args, output4, target)
    lossTotal = np.append(lossTotal, [loss1.item(), loss2.item(), loss3.item(), lossf.item(), loss0_4.item()])

    total_loss = loss0_1 + loss0_2 + loss0_3 + loss0_4

    #epoch_loss.append(total_loss)
    time_taken = time.time() - start_time

    print('[%d/%d] loss1-0_1: %.6f loss2-0_1: %.6f loss3-0_1: %.6f lossf-0_1: %.6f loss0_1: %.6f\n'
          'loss1-0_2: %.6f loss2-0_2: %.6f loss3-0_2: %.6f lossf-0_2: %.6f loss0_2: %.6f\n'
          'loss1-0_3: %.6f loss2-0_3: %.6f loss3-0_3: %.6f lossf-0_3: %.6f loss0_3: %.6f\n'
          'loss1-0_4: %.6f loss2-0_4: %.6f loss3-0_4: %.6f lossf-0_4: %.6f loss0_4: %.6f\n'
          'total_loss: %.6f time:%.2f' % (
              i, total_batches, lossTotal[0], lossTotal[1], lossTotal[2], lossTotal[3], lossTotal[4],
              lossTotal[5], lossTotal[6], lossTotal[7], lossTotal[8], lossTotal[9],
              lossTotal[10], lossTotal[11], lossTotal[12], lossTotal[13], lossTotal[14],
              lossTotal[15], lossTotal[16], lossTotal[17], lossTotal[18], lossTotal[19],
              total_loss, time_taken))
    return lossTotal, total_loss

def lossArch5(args, output1, output2, output3, output4, output5, target, i, total_batches, start_time):

    #output1, output2, output3, output4, output5 = model(input)  # output = [output1, output2, output3, output4]

    loss1, loss2, loss3, lossf, loss0_1 = lossFunc(args, output1, target)
    lossTotal = [loss1.item(), loss2.item(), loss3.item(), lossf.item(), loss0_1.item()]

    loss1, loss2, loss3, lossf, loss0_2 = lossFunc(args, output2, target)
    lossTotal = np.append(lossTotal, [loss1.item(), loss2.item(), loss3.item(), lossf.item(), loss0_2.item()])

    loss1, loss2, loss3, lossf, loss0_3 = lossFunc(args, output3, target)
    lossTotal = np.append(lossTotal, [loss1.item(), loss2.item(), loss3.item(), lossf.item(), loss0_3.item()])

    loss1, loss2, loss3, lossf, loss0_4 = lossFunc(args, output4, target)
    lossTotal = np.append(lossTotal, [loss1.item(), loss2.item(), loss3.item(), lossf.item(), loss0_4.item()])

    loss1, loss2, loss3, lossf, loss0_5 = lossFunc(args, output5, target)
    lossTotal = np.append(lossTotal, [loss1.item(), loss2.item(), loss3.item(), lossf.item(), loss0_5.item()])

    total_loss = loss0_1 + loss0_2 + loss0_3 + loss0_4 + loss0_5

    #epoch_loss.append(total_loss)
    time_taken = time.time() - start_time

    print('[%d/%d] loss1-0_1: %.6f loss2-0_1: %.6f loss3-0_1: %.6f lossf-0_1: %.6f loss0_1: %.6f\n'
          'loss1-0_2: %.6f loss2-0_2: %.6f loss3-0_2: %.6f lossf-0_2: %.6f loss0_2: %.6f\n'
          'loss1-0_3: %.6f loss2-0_3: %.6f loss3-0_3: %.6f lossf-0_3: %.6f loss0_3: %.6f\n'
          'loss1-0_4: %.6f loss2-0_4: %.6f loss3-0_4: %.6f lossf-0_4: %.6f loss0_4: %.6f\n'
          'loss1-0_5: %.6f loss2-0_5: %.6f loss3-0_5: %.6f lossf-0_4: %.6f loss0_5: %.6f\n'
          'total_loss: %.6f time:%.2f' % (
              i, total_batches, lossTotal[0], lossTotal[1], lossTotal[2], lossTotal[3], lossTotal[4],
              lossTotal[5], lossTotal[6], lossTotal[7], lossTotal[8], lossTotal[9],
              lossTotal[10], lossTotal[11], lossTotal[12], lossTotal[13], lossTotal[14],
              lossTotal[15], lossTotal[16], lossTotal[17], lossTotal[18], lossTotal[19],
              lossTotal[20], lossTotal[21], lossTotal[22], lossTotal[23], lossTotal[24], total_loss, time_taken))
    return lossTotal, total_loss
