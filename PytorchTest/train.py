from __future__ import division
from __future__ import print_function
from tapnet import load_raw_ts

import math
import sys
import time
import argparse  # argparse是一个Python模块：命令行选项、参数和子命令解析器

import numpy as np
import torch
import torch.optim as optim
from keras.metrics import accuracy

from Model import Model_4
from util import *
import torch.nn.functional as F

import seaborn as sns
from sklearn import metrics
from matplotlib import font_manager as fm, rcParams
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置

datasets = ["NATOPS", "ECG", "Heartbeat", "AtrialFibrillation", "FingerMovements", "ECG200", "ECG5000", "MIMIC-III",
            "MI", "HAR"]

parser = argparse.ArgumentParser()  # 建立解析对象

# dataset settings
parser.add_argument('--data_path', type=str, default=r"C:\Users\13091\Desktop\Graduation_Design_Resource\dataset\mit-bih-arrhythmia-database-1.0.0",
                    help='the path of data.')  # 增加属性：给xx实例增加一个aa属性 # xx.add_argument(“aa”)，在 add_argument 前，给属性名之前加上“- -”，就能将之变为可选参数
parser.add_argument('--use_muse', action='store_true', default=False,
                    help='whether to use the raw data. Default:False')
parser.add_argument('--dataset', type=str, default="NATOPS",  # Heartbeat
                    help='time series dataset. Options: See the datasets list')

# cuda settings
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# Training parameter settings
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='Initial learning rate. default:[0.00001]')
parser.add_argument('--wd', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters). default: 5e-3')
parser.add_argument('--stop_thres', type=float, default=1e-9,
                    help='The stop threshold for the training error. If the difference between training losses '
                         'between epoches are less than the threshold, the training will be stopped. Default:1e-9')

# Model parameters
parser.add_argument('--use_lstm', type=str, default=True,
                    help='whether to use LSTM for feature extraction. Default:False')
parser.add_argument('--use_cnn', type=str, default=True,
                    help='whether to use CNN for feature extraction. Default:False')
parser.add_argument('--use_rp', type=str, default=True,
                    help='Whether to use random projection')
parser.add_argument('--rp_params', type=str, default='-1,3',
                    help='Parameters for random projection: number of random projection, '
                         'sub-dimension for each random projection')
parser.add_argument('--use_metric', action='store_true', default=True,
                    help='whether to use the metric learning for class representation. Default:False')
parser.add_argument('--metric_param', type=float, default=0.000001,
                    help='Metric parameter for prototype distances between classes. Default:0.000001')
parser.add_argument('--use_ss', action='store_true', default=False,
                    help='Use semi-supervised learning.')
parser.add_argument('--filters', type=str, default="256,256,128",
                    help='filters used for convolutional network. Default:256,256,128')
parser.add_argument('--kernels', type=str, default="8,5,3",
                    help='kernels used for convolutional network. Default:8,5,3')
parser.add_argument('--dilation', type=int, default=1,
                    help='the dilation used for the first convolutional layer. '
                         'If set to -1, use the automatic number. Default:-1')
parser.add_argument('--layers', type=str, default="500,300",
                    help='layer settings of mapping function. [Default]: 500,300')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability). Default:0.5')

args = parser.parse_args()  # 属性给与args实例：把parser中设置的所有"add_argument"给返回到args子类实例当中，那么parser中增加的属性内容都会在args实例中，使用即可
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
args.sparse = True
args.layers = [int(l) for l in args.layers.split(",")]  # default="500,300"
args.kernels = [int(l) for l in args.kernels.split(",")]  # default="8,5,3"
args.filters = [int(l) for l in args.filters.split(",")]  # default="256,256,128"
args.rp_params = [int(l) for l in args.rp_params.split(",")]  # default='-1,3'

if not args.use_lstm and not args.use_cnn:
    print("Must specify one encoding method: --use_lstm or --use_cnn")
    print("Program Exiting.")
    exit(-1)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
print("Loading dataset", args.dataset, "...")
# Model and optimizer
model_type = "Model_4"  # Options: FGCN, ProtoGCN, BiGCN, MotifGCN, InterGCN, TPNet, TapNet
if model_type == "Model_4":

    # if args.use_muse:  # default=False, help='whether to use the raw data. Default:False'
    features, labels, idx_train, idx_val, idx_test, nclass \
        = load_raw_ts(args.data_path, dataset=args.dataset)

    print(features.shape)
    # else:
    #     features, labels, idx_train, idx_val, idx_test, nclass \
    #                                 = load_raw_ts(args.data_path, dataset=args.dataset)

    # features, labels, idx_train, idx_val, idx_test, nclass = load_muse(args.data_path, dataset=args.dataset, sparse=True)

    # update random permutation parameter
    if args.rp_params[0] < 0:
        # dim = features.shape[1]
        # if dim <= 6:
        #     args.rp_params = [dim, math.ceil(dim / 2)]
        # elif dim > 6 and dim <= 20:
        #     args.rp_params = [10, 3]
        # else:
        #     args.rp_params = [int(dim / 2), 3]
        dim = features.shape[1]
        args.rp_params = [3, math.floor(dim * 1.5 / 3)]

    print("rp_params:", args.rp_params)

    # update dilation parameter
    if args.dilation == -1:  # default=1, help='the dilation used for the first convolutional layer. '
        args.dilation = math.floor(features.shape[2] / 64)

    print("Data shape:", features.size())
    model = Model_4(nfeat=features.shape[1],
                    len_ts=features.shape[2],
                    layers=args.layers,
                    nclass=nclass,
                    dropout=args.dropout,
                    use_lstm=args.use_lstm,
                    use_cnn=args.use_cnn,
                    filters=args.filters,
                    dilation=args.dilation,
                    kernels=args.kernels,
                    use_ss=args.use_ss,
                    use_metric=args.use_metric,
                    use_rp=args.use_rp,
                    rp_params=args.rp_params
                    )  # def __init__(self, nfeat, len_ts, nclass, dropout, filters, kernels, dilation, layers, use_rp, rp_params,
    #              use_att=True, use_ss=False, use_metric=False, use_muse=False, use_lstm=False, use_cnn=True)

    # cuda
    if args.cuda:
        model.cuda()
        features, labels, idx_train = features.cuda(), labels.cuda(), idx_train.cuda()
    input = (features, labels, idx_train, idx_val, idx_test)

# init the optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.wd)  # 权重衰减


# training function
def train():
    loss_list = [sys.maxsize]
    test_best_possible, best_so_far = 0.0, sys.maxsize
    for epoch in range(args.epochs):  # default=100

        t = time.time()
        model.train()
        optimizer.zero_grad()

        # new_input = (features[idx_train, ], labels[idx_train], idx_train, idx_val, idx_test)
        output, proto_dist = model(input)  # input包括 train val test; proto_dist 原型距离

        loss_train = F.cross_entropy(output[idx_train], torch.squeeze(labels[idx_train]))
        if args.use_metric:
            loss_train = loss_train - args.metric_param * proto_dist

        if abs(loss_train.item() - loss_list[-1]) < args.stop_thres \
                or loss_train.item() > loss_list[-1]:
            break  # loss_train.item 比之前的损失中最后一项 还要大
        else:
            loss_list.append(loss_train.item())  # 加入表损失是在减小的

        acc_train = accuracy(output[idx_train], labels[idx_train])

        # sns.set()
        # f, ax = plt.subplots()
        # Confusion_metrix = confusion_metrix(output[idx_train], labels[idx_train])
        # sns.heatmap(Confusion_metrix, annot=True, ax=ax)  # 画热力图
        # ax.set_title('confusion matrix')  # 标题
        # ax.set_xlabel('predict')  # x轴
        # ax.set_ylabel('true')  # y轴
        # plt.savefig(str(epoch)+"_confusion.jpg")
        # plt.show()

        # plt.matshow(Confusion_metrix)
        # plt.title("混淆矩阵")
        # plt.colorbar()
        # plt.ylabel("实际类型")
        # plt.xlabel("预测类型")
        # plt.show()

        # false_positive_rate, recall, thresholds, roc_train = roc(output[idx_train], labels[idx_train])
        #
        # ax3 = plt.subplot(223)
        # ax3.set_title("Receiver Operating Characteristic", verticalalignment='center')
        # plt.plot(false_positive_rate, recall, 'b', label='AUC=%0.2f' % roc_train)
        # plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.0])
        # plt.ylabel('Recall')
        # plt.xlabel('false_positive_rate')
        # plt.savefig(str(epoch) + "_roc.jpg")
        # plt.show()

        # precision, recall, _thresholds, average_precision, pr_train = pr(output[idx_train], labels[idx_train])
        #
        # ax2 = plt.subplot(224)
        # ax2.set_title("Precision_Recall Curve PR=%0.2f" % pr_train, verticalalignment='center')
        # plt.step(precision, recall, where='post', alpha=0.2, color='r')
        # plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.ylabel('Precision')
        # plt.xlabel('Recall')
        # plt.savefig(str(epoch) + "_pr.jpg")
        # plt.show()
        loss_train.backward()
        optimizer.step()

        # if not args.fastmode:
        #     # Evaluate validation set performance separately,
        #     # deactivates dropout during validation run.
        #     model.eval()
        #     output = model(features)

        # print(output[idx_val])
        loss_val = F.cross_entropy(output[idx_val], torch.squeeze(labels[idx_val]))
        acc_val = accuracy(output[idx_val], labels[idx_val])
        # false_positive_rate_val, recall_val, thresholds_val, roc_val = roc(output[idx_val], labels[idx_val])
        # precision_val, recall_val, _thresholds_val, average_precision_val, pr_val = pr(output[idx_val], labels[idx_val])
        # # print(output[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.8f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              # 'roc_train: {:.4f}'.format(roc_train.item()),
              # 'pr_train: {:.4f}'.format(pr_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              # 'roc_val: {:.4f}'.format(roc_val.item()),
              # 'pr_val: {:.4f}'.format(pr_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        if acc_val.item() > test_best_possible:
            test_best_possible = acc_val.item()  # accuracy_val
        if best_so_far > loss_train.item():
            best_so_far = loss_train.item()  # loss_train
            test_acc = acc_val.item()  # accuracy_val
    print("test_acc: " + str(test_acc))
    print("best possible: " + str(test_best_possible))


# test function
def test():
    output, proto_dist = model(input)
    print("output", output)
    print("label", labels)
    # print(output[idx_test])
    loss_test = F.cross_entropy(output[idx_test], torch.squeeze(labels[idx_test]))
    if args.use_metric:
        loss_test = loss_test - args.metric_param * proto_dist

    acc_test = accuracy(output[idx_test], labels[idx_test])
    # false_positive_rate_test, recall_test, thresholds_test, roc_test = roc(output[idx_test], labels[idx_test])
    # precision_test, recall_test, _thresholds_test, average_precision_test, pr_test = pr(output[idx_test], labels[idx_test])
    print(args.dataset, "Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    # "roc_test= {:.4f}".format(roc_test.item()),
    # "pr_test= {:.4f}".format(pr_test.item()))


# Train model
t_total = time.time()
train()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
