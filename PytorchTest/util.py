from ABSOLUTE_PATH import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as dill
import sys
import os
import random

import torch
from tqdm import tqdm
from collections import OrderedDict, Counter

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import scipy.io
from scipy.signal import butter, lfilter, periodogram

import matplotlib.pyplot as plt


def generate_csvFile(count):
    """
    before running this data preparing code,
    transform htmlTable to .csvFile
    just run this once

    """
    html_url = default_path + '/mitdbdir/tables.htm'
    if count == 0:
        table_label = pd.read_html(html_url)[count]
        table_label.to_csv(default_path + '/label.csv', encoding='utf-8')
        # 将目录下的dat文件转换成csv文件，后续通过panda读取csv文件
        filenames = pd.read_csv(os.path.join(default_path, 'RECORDS'), header=None)
        filenames = filenames.iloc[:, 0].values
        for filename in tqdm(filenames):
            table_data = pd.read_excel




def preprocess_physionet(data_path):
    """
    before running this data preparing code, 
    please first download the raw data from https://physionet.org/content/challenge-2017/1.0.0/, 
    and put it in data_path
    """

    # read label
    label_df = pd.read_csv(os.path.join(data_path, 'label.csv'), header=None)
    label = label_df.iloc[:, 1].values
    print(Counter(label))

    # read data
    all_data = []
    filenames = pd.read_csv(os.path.join(data_path, 'RECORDS'), header=None)
    filenames = filenames.iloc[:, 0].values
    print(filenames)
    for filename in tqdm(filenames):
        # mat = scipy.io.loadmat(os.path.join(data_path, '{0}.mat'.format(filename)))
        mat = pd.read_excel(default_path + '/{0}.csv'.format(filename))
        mat = np.array(mat['val'])[0]
        all_data.append(mat)
    all_data = np.array(all_data)

    # 生成pkl文件
    res = {'data': all_data, 'label': label}
    with open(os.path.join('challenge2017.pkl'), 'wb') as fout:
        dill.dump(res, fout)


# FIR滤波器组，将单导联心电信号转换成多通道信号
def filter_channel(x):
    signal_freq = 300

    ### candidate channels for ECG
    P_wave = (0.67, 5)  # P波段大致频率范围（Hz）
    QRS_complex = (10, 50)  # QRS复合波段大致频率范围
    T_wave = (1, 7)  # T波段大致频率范围
    muscle = (5, 50)  # 肌电噪声波形大致频率范围
    resp = (0.12, 0.5)  # 呼吸波形大致频率范围
    # 综上将心电信号分为三条通道（低，中，高）并于原信号构成四通道心电信号
    ECG_preprocessed = (0.5, 50)
    wander = (0.001, 0.5)
    noise = 50

    ### use low (wander), middle (ECG_preprocessed) and high (noise) for example
    bandpass_list = [wander, ECG_preprocessed]  # 带通滤波频率范围
    highpass_list = [noise]  # 高通滤波频率范围

    nyquist_freq = 0.5 * signal_freq  # 奈奎斯特频率为原始采样频率的一半
    filter_order = 1  # 滤波器阶数，指过滤谐波的次数
    ### out including original x
    out_list = [x]

    # 带通滤波器
    for bandpass in bandpass_list:
        # 使用奈奎斯特频率作为截止频率进行归一化
        low = bandpass[0] / nyquist_freq
        high = bandpass[1] / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, x)  # 真正滤波
        out_list.append(y)  # y_1:0.001~0.5Hz, y_2:0.5~50Hz

    # 高通滤波器
    for highpass in highpass_list:
        high = highpass / nyquist_freq
        b, a = butter(filter_order, high, btype="high")  # b, a分别代表分子和分母
        y = lfilter(b, a, x)
        out_list.append(y)  # y:50+Hz

    out = np.array(out_list)
    return out  # 转换后的四通道心电信号, [x, low, middle, high] [x, y_1:0.001~0.5Hz, y_2:0.5~50Hz, y:50+Hz]


def slide_and_cut(X, Y, window_size, stride, output_pid=False):
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]
        if tmp_Y == 0:  # 非房颤 500 切分
            i_stride = stride
        elif tmp_Y == 1:  # 房颤   50  切分
            i_stride = stride // 10
        for j in range(0, len(tmp_ts) - window_size, i_stride):  # 滑动窗口信号分割成等长片段
            out_X.append(tmp_ts[j:j + window_size])
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)


def compute_beat(X):  # 计算一阶差分，依照差分后数值大小以寻找到波形急剧变化的点
    out = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for i in tqdm(range(out.shape[0]), desc="compute_beat"):
        for j in range(out.shape[1]):
            out[i, j] = np.concatenate([[0], np.abs(np.diff(X[i, j, :]))])  # np.diff才是一阶差分计算函数
    return out


def compute_rhythm(X, n_split):  # X_train:[155537, 4, 3000]
    cnt_split = int(X.shape[2] / n_split)
    out = np.zeros((X.shape[0], X.shape[1], cnt_split))
    for i in tqdm(range(out.shape[0]), desc="compute_rhythm"):
        for j in range(out.shape[1]):
            tmp_ts = X[i, j, :]
            tmp_ts_cut = np.split(tmp_ts, X.shape[2] / n_split)  # 对于训练集将每段3000分割成60个更小片段
            for k in range(cnt_split):
                out[i, j, k] = np.std(tmp_ts_cut[k])  # 计算标准差
    return out  # out:[batch, 4, 60]  155537, 4, 60


def compute_freq(X):  # X_train:[155537, 4, 3000]
    out = np.zeros((X.shape[0], X.shape[1], 1))
    fs = 300
    for i in tqdm(range(out.shape[0]), desc="compute_freq"):
        for j in range(out.shape[1]):
            _, Pxx_den = periodogram(X[i, j, :], fs)  # 使用功率谱密度函数计算每条通道功率，功率越大其能量也就越大
            out[i, j, 0] = np.sum(Pxx_den)  # 计算心电信号在该通道上的总功率
    return out  # [155537, 4, 1]


def make_data_physionet(data_path, window_size=3000, stride=500):
    # read pkl
    with open(os.path.join(data_path, 'challenge2017.pkl'), 'rb') as fin:
        res = dill.load(fin)
    ## scale data
    all_data = res['data']
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std  # normalize
    all_data = res['data']
    all_data = np.array(all_data)
    ## encode label
    all_label = []
    for i in res['label']:
        if i == 'A':
            all_label.append(1)
        else:
            all_label.append(0)
    all_label = np.array(all_label)

    # split train test
    n_sample = len(all_label)
    split_idx_1 = int(0.75 * n_sample)
    split_idx_2 = int(0.85 * n_sample)

    shuffle_idx = np.random.permutation(n_sample)  # permutation会在不打乱原始输入的基础上，返回一个无序副本
    all_data = all_data[shuffle_idx]
    all_label = all_label[shuffle_idx]

    X_train = all_data[:split_idx_1]
    X_val = all_data[split_idx_1:split_idx_2]
    X_test = all_data[split_idx_2:]
    Y_train = all_label[:split_idx_1]
    Y_val = all_label[split_idx_1:split_idx_2]
    Y_test = all_label[split_idx_2:]

    # slide and cut
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=window_size, stride=stride)
    X_val, Y_val = slide_and_cut(X_val, Y_val, window_size=window_size, stride=stride)
    X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride, output_pid=True)
    print('after: ')
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))

    # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    # multi-level
    X_train_ml = []
    X_val_ml = []
    X_test_ml = []
    for i in tqdm(X_train, desc="X_train_ml"):  # tqdm 进度条
        tmp = filter_channel(i)  # 单导联心电信号 -> 多通道心电信号 [x, y_1:0.001~0.5Hz, y_2:0.5~50Hz, y:50+Hz]
        X_train_ml.append(tmp)
    X_train_ml = np.array(X_train_ml)
    for i in tqdm(X_val, desc="X_val_ml"):
        tmp = filter_channel(i)
        X_val_ml.append(tmp)
    X_val_ml = np.array(X_val_ml)
    for i in tqdm(X_test, desc="X_test_ml"):
        tmp = filter_channel(i)
        X_test_ml.append(tmp)
    X_test_ml = np.array(X_test_ml)
    print(X_train_ml.shape, X_val_ml.shape, X_test_ml.shape)

    # save
    res = {'Y_train': Y_train, 'Y_val': Y_val, 'Y_test': Y_test, 'pid_test': pid_test}
    with open(os.path.join(data_path, 'ecgnet_info.pkl'), 'wb') as fout:  # 生成标签文件
        dill.dump(res, fout)

    fout = open(os.path.join(data_path, 'ecgnet_X_train.bin'), 'wb')
    np.save(fout, X_train_ml)
    fout.close()

    fout = open(os.path.join(data_path, 'ecgnet_X_val.bin'), 'wb')
    np.save(fout, X_val_ml)
    fout.close()

    fout = open(os.path.join(data_path, 'ecgnet_X_test.bin'), 'wb')
    np.save(fout, X_test_ml)
    fout.close()


def make_knowledge_physionet(data_path, n_split=50):  # 3000， 500， 50

    # read
    fin = open(os.path.join(data_path, 'ecgnet_X_train.bin'), 'rb')  # 调用  make_data_physionet 生成的4个文件
    X_train = np.load(fin)
    fin.close()
    fin = open(os.path.join(data_path, 'ecgnet_X_val.bin'), 'rb')
    X_val = np.load(fin)
    fin.close()
    fin = open(os.path.join(data_path, 'ecgnet_X_test.bin'), 'rb')
    X_test = np.load(fin)
    fin.close()

    # compute knowledge
    K_train_beat = compute_beat(X_train)  # 样本长度3000
    K_train_rhythm = compute_rhythm(X_train, n_split)
    K_train_freq = compute_freq(X_train)

    K_val_beat = compute_beat(X_val)
    K_val_rhythm = compute_rhythm(X_val, n_split)
    K_val_freq = compute_freq(X_val)

    K_test_beat = compute_beat(X_test)
    K_test_rhythm = compute_rhythm(X_test, n_split)
    K_test_freq = compute_freq(X_test)

    # save
    fout = open(os.path.join(data_path, 'ecgnet_K_train_beat.bin'), 'wb')
    np.save(fout, K_train_beat)
    fout.close()
    fout = open(os.path.join(data_path, 'ecgnet_K_val_beat.bin'), 'wb')
    np.save(fout, K_val_beat)
    fout.close()
    fout = open(os.path.join(data_path, 'ecgnet_K_test_beat.bin'), 'wb')
    np.save(fout, K_test_beat)
    fout.close()

    res = {'K_train_rhythm': K_train_rhythm, 'K_train_freq': K_train_freq,
           'K_val_rhythm': K_val_rhythm, 'K_val_freq': K_val_freq,
           'K_test_rhythm': K_test_rhythm, 'K_test_freq': K_test_freq}
    with open(os.path.join(data_path, 'ecgnet_knowledge.pkl'), 'wb') as fout:
        dill.dump(res, fout)


def evaluate(gt, pred):
    '''
    gt 是心电信号标签
    pred 是模型输出的预测值
    '''

    pred_label = []
    for i in pred:
        pred_label.append(np.argmax(i))
    pred_label = np.array(pred_label)

    res = OrderedDict({})

    res['auroc'] = roc_auc_score(gt, pred[:, 1])
    res['auprc'] = average_precision_score(gt, pred[:, 1])
    res['f1'] = f1_score(gt, pred_label)

    res['\nmat'] = confusion_matrix(gt, pred_label)  # 混淆矩阵

    for k, v in res.items():
        print(k, ':', v, '|', end='')
    print()

    return list(res.values())


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.now_score = None
        self.val_loss_min = np.Inf
        self.val_loss_now = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.now_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.now_score + self.delta:
            self.counter += 1
            print(f'Validation loss increased ({self.val_loss_now:.6f} --> {val_loss:.6f}).')
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.val_loss_now = val_loss
            self.now_score = score
            if self.counter >= self.patience:
                self.early_stop = True
        elif score < self.best_score + self.delta:
            self.now_score = score
            print(f'Validation loss decreased ({self.val_loss_now:.6f} --> {val_loss:.6f}).')
            self.val_loss_now = val_loss
            self.counter = 0
        else:
            self.best_score = score
            self.now_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Minimum Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, '{}/model/best_model.pt'.format(path))  # 这里会存储迄今为止最优的模型
        self.val_loss_min = val_loss
        self.val_loss_now = val_loss
