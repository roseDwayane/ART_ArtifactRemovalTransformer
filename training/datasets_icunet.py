import csv
import os
import random
import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset
import pickle
from training.utils import partial_channel2

epsilon = np.finfo(float).eps

class myDataset(Dataset):
    def __init__(self, mode, iter=20, data="0-3", train_len=0, block_num=1):
        self.sample_rate = 256
        self.lenth = train_len
        self.lenthtest = 3600
        self.lenthval = 3500
        self.mode = mode
        self.iter = iter
        self.savedata = data
        self.block_num = block_num

    def __len__(self):
        if self.mode == 2:
            return self.lenthval
        elif self.mode == 1:
            return self.lenthtest
        else:
            return self.lenth

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        data_mode = ["Brain", "ChannelNoise", "Eye", "Heart", "LineNoise", "Muscle", "Other"]

        if self.mode == 2:
            allFileList = os.listdir("./Real_EEG/val/Brain/")
            file_name = './Real_EEG/val/Brain/' + allFileList[idx]
            data_clean = self.read_train_data(file_name)
            for i in range(7):
                file_name = './Real_EEG/val/' + data_mode[random.randint(0, 6)] + '/' + allFileList[idx]
                if os.path.isfile(file_name):
                    data_nosie = self.read_train_data(file_name)
                    break
                else:
                    data_nosie = data_clean
        elif self.mode == 1:
            allFileList = os.listdir("./Real_EEG/test/Brain/")
            file_name = './Real_EEG/test/Brain/' + allFileList[idx]
            data_clean = self.read_train_data(file_name)
            for i in range(7):
                file_name = './Real_EEG/test/' + data_mode[random.randint(0, 6)] + '/' + allFileList[idx]
                if os.path.isfile(file_name):
                    data_nosie = self.read_train_data(file_name)
                    break
                else:
                    data_nosie = data_clean
        else:
            allFileList = os.listdir("./Real_EEG/train/Brain/")
            file_name = './Real_EEG/train/Brain/' + allFileList[idx]
            #print("dataloader: ", file_name)
            data_clean = self.read_train_data(file_name)
            for i in range(7):
                file_name = './Real_EEG/train/' + data_mode[random.randint(0, 6)] + '/' + allFileList[idx]
                if os.path.isfile(file_name):
                    data_nosie = self.read_train_data(file_name)
                    break
                else:
                    data_nosie = data_clean
        #print(file_name)



        #print("data_set", noise.shape)

        max_num = np.max(data_nosie)
        data_avg = np.average(data_nosie)
        data_std = np.std(data_nosie)
        #max_num = 100
        #print("max_num: ", max_num)

        #target = np.array(data / max_num).astype(np.float)
        if int(data_std) != 0:
            target = np.array((data_clean - data_avg) / data_std).astype(np.float64)
            attr   = np.array((data_nosie - data_avg) / data_std).astype(np.float64)
        else:
            target = np.array(data_clean - data_avg).astype(np.float64)
            attr   = np.array(data_nosie - data_avg).astype(np.float64)

        ## partial channel
        if self.block_num:
            target, attr = partial_channel2(target, attr, 22)

        target = target.copy()
        target = torch.tensor(target, dtype=torch.float32)

        attr = attr.copy()
        attr = torch.tensor(attr, dtype=torch.float32)

        return attr, target, data_std

    def my_collate(batch):
        data, targets = list(), list()
        for b in batch:
            data.append(b[0])
            targets.append(b[1])
        data = torch.stack(data, dim=0)
        targets = torch.stack(targets, dim=0)
        return data, targets

    def read_simulate_data(self, file_name):
        with open(file_name, 'rb+') as f:
            data = pickle.load(f)
        data = np.array(data).astype(np.float64)
        return data

    def read_train_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        new_data = np.array(data).astype(np.float64)

        ''' for training 19 channels
        row = np.array([0,1,2,3,4,5,6,12,13,14,15,16,22,23,24,25,26,27,29])
        new_data = []
        for i in range(19):
            #print(i, row[i])
            #print(data[row[i]].shape)
            new_data.append(data[row[i]])
            new_data = np.array(new_data).astype(np.float64)
        '''

        #data = data.T
        return new_data

    def read_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float64)
        #data = data.T
        return data

    def lowpass60hz(self, data, sample_rate):
        b, a = signal.butter(8, 2 * 60 / sample_rate, 'lowpass')  # 0.48 = 2*hz/sample_rate
        data = signal.filtfilt(b, a, data)  # data 要過濾的波
        return data

    def peak_gen(self, sample_rate):
        # peak --------------------------------
        x = random.randint(64, 512)
        while x % 2 == 0:
            x = random.randint(64, 512)
        y = random.uniform(1.5, 2.0)
        x1 = np.linspace(0, (x // 2 - 1), x // 2)
        x2 = np.linspace((x // 2), x, x // 2)
        y1 = y * x1
        y2 = -y * x2 + x * y
        peak = np.concatenate([y1, y2])
        X = np.linspace(0, 4, sample_rate)
        a = random.randint(1, sample_rate - x)
        peak = np.pad(peak, (a, sample_rate - x - a + 1), 'constant')
        # random noise ------------------------
        rand_arr = np.random.randn(sample_rate)
        # sin ---------------------------------
        x_sin = np.linspace(np.random.randn() * 10, np.random.randn(), sample_rate)
        sin_arr = np.sin(x_sin)

        noise = peak + 0.1 * rand_arr + 0.1 * sin_arr
        # plt.plot(X, noise)
        # plt.xlabel('Time [sec]')
        # plt.ylabel('Amplitude [µ]')
        # plt.show()
        return noise
