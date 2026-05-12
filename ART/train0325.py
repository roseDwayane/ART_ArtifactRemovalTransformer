import os
import torch
import pickle
#import Models.tiny_model as tiny_model
import cumbersome_model
#from Criteria import CrossEntropyLoss2d, FocalLoss
import UNet_family
import UNet_attention
import torch.nn as nn
import torch.backends.cudnn as cudnn
from myDataset import myDataset
from loss_func import lossFunc, lossArch3, lossArch4, lossArch5
import time
import torch.optim.lr_scheduler
import numpy as np
from scipy import signal
from utils import draw_raw, imgSave, numpy_SNR, draw_psd, SNR_cal
#from dataGenerator import dataInit, dataDelete, dataRestore
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def SNR(args, val_loader, model, epoch):
    model.eval()
    for i , (input, target, max_num) in enumerate(val_loader):
        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()

        if args.model == "UNet_family":
            with torch.no_grad():
                # run the model
                decode1, decode2, decode3, decode4 = model(input)

        elif args.model == "cumbersome_model":
            with torch.no_grad():
                decode4 = model(input)

        i_t, i_d = SNR_cal(input, target, decode4, max_num)
        tmean = np.nanmean(i_t)
        tstd = np.nanstd(i_t)
        dmean = np.nanmean(i_d)
        dstd = np.nanstd(i_d)

        print("SNR(shap): ", i_d.shape)
        #print(np.nanmean(i_t))
        print(tmean, tstd, dmean, dstd)
        break
    return tmean, tstd, dmean, dstd


def draw_sub(args, val_loader, model, epoch):
    Channel_location = ["FP1", "FP2",
                        "F7", "F3", "FZ", "F4", "F8",
                        "T7", "C3", "CZ", "C4", "T8",
                        "P7", "P3", "PZ", "P4", "P8",
                        "O1", "O2"]

    model.eval()
    for i, (input, target, max_num) in enumerate(val_loader):
        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()
        with torch.no_grad():
            # run the mdoel
            decode1, decode2, decode3, decode4 = model(input)
        batch, channel = 0, 14
        print("decode(shape): ", decode4.shape)
        for j in range(19):
            filename = Channel_location[j] + 'e{}'.format(str(str(epoch)))
            snr_i_t = numpy_SNR(input[batch][j], target[batch][j])
            snr_i_d = numpy_SNR(input[batch][j], decode4[batch][j])
            draw_psd(filename, input[batch][j], 256, snr_i_t, snr_i_d, 1, 2, 1)
            draw_psd(filename, decode4[batch][j], 256, snr_i_t, snr_i_d, 1, 2, 2)
            draw_psd(filename, target[batch][j], 256, snr_i_t, snr_i_d, 1, 2, 3)

            imgSave(args.savefig + "/sub/",  str(j) + "_" + filename)

        loss4one = []
        loss4sec = []
        loss4third = []
        criterion = nn.MSELoss()
        for j in range(int(decode4.shape[1])):
            loss = criterion(decode4[:, j, :], target[:, j, :])

            doutput = decode4[:, j, 1:] - decode4[:, j, :-1]
            dtarget = target[:, j, 1:] - target[:, j, :-1]
            loss2 = criterion(doutput, dtarget)

            # print("train(shape):", doutput.shape)

            d2output = doutput[:, 1:] - doutput[:, :-1]
            d2target = dtarget[:, 1:] - dtarget[:, :-1]
            loss3 = criterion(d2output, d2target)

            loss4one.append(loss.item())
            loss4sec.append(loss2.item())
            loss4third.append(loss3.item())

        fp = open(args.savefig + "/sub/" + "Channel_loss.txt", "a")
        fp.write("MSE(one): " + str(loss4one))
        fp.write("\nMSE(sec): " + str(loss4sec))
        fp.write("\nMSE(Third): " + str(loss4third))
        fp.close()
        print("MSE(one): ", loss4one)
        print("MSE(sec): ", loss4sec)
        print("MSE(Third): ", loss4third)


def draw(args, val_loader, model, epoch):
    '''
    :param args: general arguments
    :param val_loader: loaded for validation dataset
    :param model: model
    :return: non
    '''
    Channel_location = [    "FP1", "FP2",
                    "F7", "F3", "FZ", "F4", "F8",
                 "FT7", "FC3", "FCZ", "FC4", "FT8",
                    "T4", "C3", "FZ", "C4", "T4",
                 "TP7", "CP3", "CPZ", "CP4", "TP8",
                    "T5", "P3", "PZ", "P4", "T6",
                          "O1", "OZ", "O2"]

    model.eval()
    for i, (input, target, max_num) in enumerate(val_loader):
        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()

        if args.model=="UNet_family":
            with torch.no_grad():
                # run the model
                decode1, decode2, decode3, decode4 = model(input)

        elif args.model == "cumbersome_model":
            with torch.no_grad():
                decode4 = model(input)

        batch, channel = 0, 14
        #print("draw(max): ", max_num[batch])
        print("draw(shape): ", input.shape)

        target = target * max_num[batch]
        snr = numpy_SNR(target[batch][channel], target[batch][channel])
        draw_raw("target", target[batch][channel], 1, snr)

        input = input * max_num[batch]
        snr = numpy_SNR(input[batch][channel], target[batch][channel])
        draw_raw("input", input[batch][channel], 2, snr)

        decode4 = decode4 * max_num[batch]
        snr = numpy_SNR(decode4[batch][channel], target[batch][channel])
        draw_raw("decode", decode4[batch][channel], 3, snr)

        #imgSave(epoch)
        snr_i_t = numpy_SNR(input[batch][channel], target[batch][channel])
        snr_i_d4 = numpy_SNR(input[batch][channel], decode4[batch][channel])
        draw_psd(str(epoch), input[batch][channel], 256, snr_i_t, snr_i_d4, 1, 2, 1)
        draw_psd(str(epoch), decode4[batch][channel], 256, snr_i_t, snr_i_d4, 1, 2, 2)
        draw_psd(str(epoch), target[batch][channel], 256, snr_i_t, snr_i_d4, 1, 2, 3)
        imgSave(args.savefig, '/e{}'.format(str(str(epoch) + "PSD")))

        break

def val(args, val_loader, model, criterion):
    '''
    :param args: general arguments
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss
    '''
    #switch to evaluation mode
    model.eval()

    epoch_loss = []

    total_batches = len(val_loader)
    for i, (input, target, max_num) in enumerate(val_loader):
        start_time = time.time()

        if args.onGPU == True:
            input = input.cuda()    #myDataset.py -> attr
            target = target.cuda()

        if args.model == 0 or args.model == 1 or args.model == 2:
            with torch.no_grad():
                output = model(input)
            loss1, loss2, loss3, lossf, total_loss = lossFunc(args, output, target)
            lossTotal = [loss1.item(), loss2.item(), loss3.item(), lossf.item(), total_loss.item()]
            epoch_loss.append(total_loss)
            time_taken = time.time() - start_time
            print('[%3d/%3d] loss1: %.8f loss2: %.8f loss3: %.8f lossf: %.8f\n'
                  'total_loss: %.8f time:%.8f'
                  % (i, total_batches, loss1.item(), loss2.item(), loss3.item(), lossf.item(),
                     total_loss.item(), time_taken))

        elif args.model == 3:
            with torch.no_grad():
                output1, output2, output3 = model(input)

            lossTotal, total_loss = lossArch3(args, output1, output2, output3, target, i, total_batches, start_time)
            epoch_loss.append(total_loss)

        elif args.model == 4:
            with torch.no_grad():
                output1, output2, output3, output4 = model(input)
            lossTotal, total_loss = lossArch4(args, output1, output2, output3, output4, target, i, total_batches, start_time)
            epoch_loss.append(total_loss)


        elif args.model == 5:
            with torch.no_grad():
                output1, output2, output3, output4, output5 = model(input)
            lossTotal, total_loss = lossArch5(args, output1, output2, output3, output4, output5, target, i, total_batches, start_time)
            epoch_loss.append(total_loss)

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)


    return lossTotal, total_loss

def train(args, train_loader, model, criterion, optimizer, epoch):
    '''
    :param args: general arguments
    :param train_loader: loaded for training dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to train mode
    model.train()

    epoch_loss = []

    total_batches = len(train_loader)
    for i, (input, target, max_num) in enumerate(train_loader):
        #print("train: ", input.shape)
        start_time = time.time()
        if args.onGPU:
            input = input.cuda()
            target = target.cuda()

        if args.model == 0 or args.model == 1 or args.model == 2:
            output = model(input)
            loss1, loss2, loss3, lossf, total_loss = lossFunc(args, output, target)

            epoch_loss.append(total_loss)
            time_taken = time.time() - start_time
            print('[%3d/%3d] loss1: %.8f loss2: %.8f loss3: %.8f lossf: %.8f\n'
                  'total_loss: %.8f time:%.8f'
                  % (i, total_batches, loss1.item(), loss2.item(), loss3.item(), lossf.item(),
                     total_loss.item(), time_taken))

        elif args.model == 3:
            output1, output2, output3 = model(input)
            lossTotal, total_loss = lossArch3(args, output1, output2, output3, target, i, total_batches, start_time)
            epoch_loss.append(total_loss)


        elif args.model == 4:
            output1, output2, output3, output4 = model(input)
            lossTotal, total_loss = lossArch4(args, output1, output2, output3, output4, target, i, total_batches, start_time)
            epoch_loss.append(total_loss)


        elif args.model == 5:
            output1, output2, output3, output4, output5 = model(input)
            lossTotal, total_loss = lossArch5(args, output1, output2, output3, output4, output5, target, i, total_batches, start_time)
            epoch_loss.append(total_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train

def save_checkpoint(state, is_best, save_path):
    """
    Save model checkpoint.
    :param state: model state
    :param is_best: is this checkpoint the best so far?
    :param save_path: the path for saving
    """
    filename = 'checkpoint.pth.tar'
    torch.save(state, os.path.join(save_path, filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join(save_path, 'BEST_' + filename))

def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

def trainValidateSegmentation(args):
    '''
    Main function for training and validation
    :param args: global arguments
    :return: None
    '''
    # check if processed data file exists or not
    if not os.path.isfile(args.loadpickle):
        print('Error while pickling data. Please checking data processing firstly.')
    #        exit(-1)
    else:
        data = pickle.load(open(args.loadpickle, "rb"))

    # load the model

    if args.model == 0:
        model = cumbersome_model.UNet0(n_channels=30, n_classes=30, bilinear=True)
    elif args.model == 1:
        model = cumbersome_model.UNet1(n_channels=30, n_classes=30, bilinear=True)
    elif args.model == 2:
        model = cumbersome_model.UNet2(n_channels=30, n_classes=30, bilinear=True)
    elif args.model == 3:
        #model = UNet_family.NestedUNet3(num_classes=30)
        model = UNet_attention.UNetpp3_Transformer(num_classes=30)
    elif args.model == 4:
        model = UNet_family.NestedUNet4(num_classes=30)
    elif args.model == 5:
        model = UNet_family.NestedUNet5(num_classes=30)

    args.savedir = args.savedir + '/'
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    if args.onGPU:
        # model = model.to(device)
        model = model.cuda()

    # create the directory if not exist
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    # define optimization criteria
    # weight = torch.from_numpy(data['classWeights']) # convert the numpy array to torch
    # weight = torch.FloatTensor([0.500001, 0.5000001]) # convert the numpy array to torch
    # if args.onGPU:
    #     weight = weight.cuda()

    criteria = nn.MSELoss()  #
    # criteria = CrossEntropyLoss2d(weight) #weight
    # criteria = FocalLoss(3, weight)

    if args.onGPU:
        criteria = criteria.cuda()

    # since we training from scratch, we create data loaders at different scales
    # so that we can generate more augmented data and prevent the network from overfitting

    #trainLoader = torch.utils.data.DataLoader(
    #    myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], (args.inWidth, args.inHeight), flag_aug=0),
    #    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    # ****#
    # if args.onGPU:
    #     cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    # ****#

    start_epoch = 0

    if args.resume:  # 當機回復
        if os.path.isfile(args.resumeLoc):
            print("=> loading checkpoint '{}'".format(args.resume))
            #checkpoint = torch.load(args.resumeLoc, map_location='cpu')
            checkpoint = torch.load(args.resumeLoc, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\t%s\t%s" % (
            'Epoch', 'Loss(Tr)', 'Loss(Ts)', 'Loss(val)', 'Loss(val1-0_1)', 'Loss(val2-0_1)', 'Loss(val3-0_1)', 'Loss(val_f-0_1)', 'Loss(val-0_1)','Loss(val1-0_2)', 'Loss(val2-0_2)', 'Loss(val3-0_2)', 'Loss(val_f-0_2)', 'Loss(val-0_2)','Loss(val1-0_3)', 'Loss(val2-0_3)', 'Loss(val3-0_3)', 'Loss(val_f-0_3)', 'Loss(val-0_3)','Loss(val1-0_4)', 'Loss(val2-0_4)', 'Loss(val3-0_4)', 'Loss(val_f-0_4)', 'Loss(val-0_4)', 'Learning_rate', "Time", "i_t_std", "i_d_mean", "i_d_std"))
    logger.flush()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    # we step the loss by 2 after step size is reached
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=0.1, last_epoch=-1)

    best_loss = 100

    for my_iter in range(1):

        trainset = myDataset(mode=0, iter=my_iter+30, data=args.savedata, train_len=args.train_len, block_num=args.block_num)  # file='./EEG_EC_Data_csv/train.txt'
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

        testset = myDataset(mode=1, iter=my_iter + 30, data=args.savedata, block_num=args.block_num)  # file='./EEG_EC_Data_csv/train.txt'
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

        # valLoader = torch.utils.data.DataLoader(
        #    myDataLoader.MyDataset(data['valIm'], data['valAnnot'], (args.inWidth, args.inHeight), flag_aug=0),
        #    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        valset = myDataset(mode=2, iter=my_iter+2, data=args.savedata, block_num=args.block_num)
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)


        for epoch in range(start_epoch, args.max_epochs):
            start_time = time.time()
            scheduler.step(epoch)
            lr = 0
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print("Learning rate: " + str(lr))

            # train for one epoch
            # We consider 1 epoch with all the training data (at different scales)

            #time.sleep(30)

            lossTr = train(args, train_loader, model, criteria, optimizer, epoch)
            #time.sleep(3)
            # evaluate on validation set
            # 改輸出檔案的LOSS
            # return of val func
            # lossTs
            lossTs, totalLossTs= val(args, test_loader, model, criteria)
            #time.sleep(1)
            lossVal, totalLossVal = val(args, val_loader, model, criteria)
            #time.sleep(1)

            #draw(args, test_loader, model, epoch)
            #draw_sub(args, train_loader, model, start_epoch)
            #draw(args, val_loader, model, epoch)
            #tmean, tstd, dmean, dstd = SNR(args, train_loader, model, epoch)
            tmean, tstd, dmean, dstd = [0, 0, 0, 0]
            #print(tmean, tstd, dmean, dstd)

            # Did validation loss improve?
            is_best = totalLossTs < best_loss
            best_loss = min(totalLossTs, best_loss)

            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            # Save checkpoint

            state = {'epoch': epoch + 1,
                     'arch': str(model),
                     'epochs_since_improvement': epochs_since_improvement,
                     'best_loss': best_loss,
                     'state_dict': model.state_dict(),
                     'lossTr': lossTr,
                     'lossTs': totalLossTs,
                     'lossVal': totalLossVal,
                     ' lr': lr}
            save_checkpoint(state, is_best, args.savedir)
            # modified %.4f\t\t * 16

            print("\nEpoch : " + str(epoch) + ' Details:')
            print("Epoch No.: %d/%d\tTrain Loss = %.8f\tVal Loss = %.8f" % (
                epoch, args.max_epochs, lossTr, totalLossVal))
            time_taken = time.time() - start_time
            print("Time: ", time_taken)

            if args.model == 0 or args.model == 1 or args.model == 2:
                logger.write("\n%d\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
                epoch, lossTr, totalLossTs, totalLossVal, lossVal[0], lossVal[1], lossVal[2], lossVal[3], lr, time_taken, tstd, dmean, dstd))

            elif args.model == 3:
                logger.write("\n%d\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
                epoch, lossTr, totalLossTs, totalLossVal, lossVal[0], lossVal[1], lossVal[2], lossVal[3], lossVal[4],
                            lossVal[5], lossVal[6], lossVal[7], lossVal[8], lossVal[9],
                            lossVal[10], lossVal[11], lossVal[12], lossVal[13], lossVal[14],
                            lr, time_taken, tstd, dmean, dstd))
            elif args.model == 4:
                logger.write("\n%d\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
                epoch, lossTr, totalLossTs, totalLossVal, lossVal[0], lossVal[1], lossVal[2], lossVal[3], lossVal[4],
                            lossVal[5], lossVal[6], lossVal[7], lossVal[8], lossVal[9],
                            lossVal[10], lossVal[11], lossVal[12], lossVal[13], lossVal[14],
                            lossVal[15], lossVal[16], lossVal[17], lossVal[18], lossVal[19], lr, time_taken, tstd, dmean, dstd))
            elif args.model == 5:
                logger.write("\n%d\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
                epoch, lossTr, totalLossTs, totalLossVal, lossVal[0], lossVal[1], lossVal[2], lossVal[3], lossVal[4],
                            lossVal[5], lossVal[6], lossVal[7], lossVal[8], lossVal[9],
                            lossVal[10], lossVal[11], lossVal[12], lossVal[13], lossVal[14],
                            lossVal[15], lossVal[16], lossVal[17], lossVal[18], lossVal[19],
                            lossVal[20], lossVal[21], lossVal[22], lossVal[23], lossVal[24], lr, time_taken, tstd, dmean, dstd))

            logger.flush()

        '''
                if os.path.isfile(args.resumeLoc):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resumeLoc, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            #draw_sub(args, train_loader, model, start_epoch)
        '''

    logger.close()

class model_train_parameter():
    def __init__(self, loss, save, data, train_len, arch_num):
        self.model = 3 #arch_num # cumbersome_model UNet_family change to UNET++
        self.block_num = arch_num
        self.max_epochs = 60
        self.num_workers = 0
        self.batch_size = 128
        self.sample_rate = 256
        self.step_loss = 100  # Decrease learning rate after how many epochs.
        self.milestones = [50, 100, 125, 140]
        self.train_len = train_len
        self.loss = loss
        self.lr = 0.01  # 'Initial learning rate'
        self.save = save
        self.savedata = data
        self.savedir = self.save + '/modelsave'  # directory to save the results
        self.savefig = self.save + '/result'
        self.resume = True  # Use this flag to load last checkpoint for training
        self.resumeLoc = self.save + '/modelsave/checkpoint.pth.tar'
        self.classes = 2  # No of classes in the dataset.
        self.logFile = 'model_trainValLog.txt'  # File that stores the training and validation logs
        self.onGPU = True  # Run on CPU or GPU. If TRUE, then GPU.
        self.pretrained = ''  # Pretrained model
        self.loadpickle = './'



def main_train():
    for i in range(14):
        #i = i+5
        name = "0329"
        #dataRestore(name)
        trainValidateSegmentation(args=model_train_parameter([1, 1, 1, 1], './' + name + '_RealEEG_1_' + str(i), "./" + name + "_simulate_data/", 39200, i))
        #trainValidateSegmentation(args=model_train_parameter([1, 1, 1, 1], './' + name + '_RealEEG_2_' + str(i), "./" + name + "_simulate_data/", 20000, i))
        #trainValidateSegmentation(args=model_train_parameter([1, 1, 1, 1], './' + name + '_RealEEG_3_' + str(i), "./" + name + "_simulate_data/", 10000, i))
        #trainValidateSegmentation(args=model_train_parameter([1, 1, 1, 1], './' + name + '_RealEEG_4_' + str(i), "./" + name + "_simulate_data/", 5000, i))
        #trainValidateSegmentation(args=model_train_parameter([1, 1, 1, 1], './' + name + '_RealEEG_5_' + str(i), "./" + name + "_simulate_data/", 2500, i))
        #dataDelete("./" + name + "_simulate_data/")

if __name__ == '__main__':
    main_train()
