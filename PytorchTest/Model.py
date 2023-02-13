import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE

from tapnet import euclidean_dist, normalize, output_conv_size, dump_embedding
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.distributions import Categorical

from sklearn.decomposition import PCA

matplotlib.use('TkAgg')


class Model_4(nn.Module):

    def __init__(self, nfeat, len_ts, nclass, dropout, filters, kernels, dilation, layers, use_rp, rp_params,
                 use_att=True, use_ss=False, use_metric=False, use_muse=False, use_lstm=False, use_cnn=True):
        super(Model_4, self).__init__()
        self.nclass = nclass
        print(self.nclass)
        self.dropout = dropout
        self.use_metric = use_metric
        self.use_muse = use_muse
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn

        # parameters for random projection  rp
        self.use_rp = use_rp
        self.rp_group, self.rp_dim = rp_params

        # LSTM
        self.channel = nfeat
        self.ts_length = len_ts

        self.lstm_dim = 128
        self.lstm = nn.LSTM(self.ts_length, self.lstm_dim)

        # compute the size of input for fully connected layers
        lstm_fc_input = 0
        lstm_fc_input += self.lstm_dim

        # Representation lstm_mapping function
        lstm_layers = [lstm_fc_input] + layers
        print("Layers", lstm_layers)

        self.lstm_mapping = nn.Sequential()
        for i in range(len(lstm_layers) - 2):
            self.lstm_mapping.add_module("fc_" + str(i), nn.Linear(lstm_layers[i], lstm_layers[i + 1]))
            self.lstm_mapping.add_module("bn_" + str(i), nn.BatchNorm1d(lstm_layers[i + 1]))
            self.lstm_mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # add last layer
        self.lstm_mapping.add_module("fc_" + str(len(lstm_layers) - 2), nn.Linear(lstm_layers[-2], lstm_layers[-1]))
        if len(lstm_layers) == 2:  # if only one layer, add batch normalization
            self.lstm_mapping.add_module("bn_" + str(len(lstm_layers) - 2), nn.BatchNorm1d(lstm_layers[-1]))

        # Attention
        lstm_att_dim, lstm_semi_att_dim = 128, 128
        self.use_att = use_att
        if self.use_att:
            self.lstm_att_models = nn.ModuleList()
            for _ in range(nclass):
                lstm_att_models = nn.Sequential(
                    nn.Linear(layers[-1], lstm_att_dim),
                    nn.Tanh(),
                    nn.Linear(lstm_att_dim, 1)
                )
                self.lstm_att_models.append(lstm_att_models)

        self.use_ss = use_ss  # whether to use semi-supervised mode
        if self.use_ss:
            self.lstm_semi_att = nn.Sequential(
                nn.Linear(layers[-1], lstm_semi_att_dim),
                nn.Tanh(),
                nn.Linear(lstm_semi_att_dim, self.nclass)
            )

        # convolutional layer
        # features for each hidden layers
        # out_channels = [256, 128, 256]
        # filters = [256, 256, 128]
        # poolings = [2, 2, 2]
        paddings = [0, 0, 0]
        print("dilation", dilation)
        if self.use_rp:
            self.conv_1_models = nn.ModuleList()
            self.idx = []
            for i in range(self.rp_group):
                self.conv_1_models.append(
                    nn.Conv1d(self.rp_dim, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1,
                              padding=paddings[0]))
                self.idx.append(np.random.permutation(nfeat)[0: self.rp_dim])
        else:
            self.conv_1 = nn.Conv1d(self.channel, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1,
                                    padding=paddings[0])
        # self.maxpool_1 = nn.MaxPool1d(poolings[0])
        self.conv_bn_1 = nn.BatchNorm1d(filters[0])

        self.conv_2 = nn.Conv1d(filters[0], filters[1], kernel_size=kernels[1], stride=1, padding=paddings[1])
        # self.maxpool_2 = nn.MaxPool1d(poolings[1])
        self.conv_bn_2 = nn.BatchNorm1d(filters[1])

        self.conv_3 = nn.Conv1d(filters[1], filters[2], kernel_size=kernels[2], stride=1, padding=paddings[2])
        # self.maxpool_3 = nn.MaxPool1d(poolings[2])
        self.conv_bn_3 = nn.BatchNorm1d(filters[2])

        # compute the size of input for fully connected layers
        cnn_fc_input = 0
        if self.use_cnn:
            conv_size = len_ts
            for i in range(len(filters)):
                conv_size = output_conv_size(conv_size, kernels[i], 1, paddings[i])
            cnn_fc_input += conv_size
        if self.use_rp:
            cnn_fc_input = self.rp_group * filters[2]

        # Representation mapping function
        cnn_layers = [cnn_fc_input] + layers
        print("Layers", cnn_layers)
        self.cnn_mapping = nn.Sequential()
        for i in range(len(cnn_layers) - 2):
            self.cnn_mapping.add_module("fc_" + str(i), nn.Linear(cnn_layers[i], cnn_layers[i + 1]))
            self.cnn_mapping.add_module("bn_" + str(i), nn.BatchNorm1d(cnn_layers[i + 1]))
            self.cnn_mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # add last layer
        self.cnn_mapping.add_module("fc_" + str(len(cnn_layers) - 2), nn.Linear(cnn_layers[-2], cnn_layers[-1]))
        if len(cnn_layers) == 2:  # if only one layer, add batch normalization
            self.cnn_mapping.add_module("bn_" + str(len(cnn_layers) - 2), nn.BatchNorm1d(cnn_layers[-1]))

        # CNN_Attention
        cnn_att_dim, cnn_semi_att_dim = 128, 128
        self.use_att = use_att
        if self.use_att:
            self.cnn_att_models = nn.ModuleList()
            for _ in range(nclass):
                cnn_att_models = nn.Sequential(
                    nn.Linear(cnn_layers[-1], cnn_att_dim),
                    nn.Tanh(),
                    nn.Linear(cnn_att_dim, 1)
                )
                self.cnn_att_models.append(cnn_att_models)

        self.use_ss = use_ss  # whether to use semi-supervised mode
        if self.use_ss:
            self.cnn_semi_att = nn.Sequential(
                nn.Linear(cnn_layers[-1], cnn_semi_att_dim),
                nn.Tanh(),
                nn.Linear(cnn_semi_att_dim, self.nclass)
            )

        # compute the size of input for fully connected layers
        fc_input = 600
        # Representation mapping function
        layers = [fc_input] + layers
        print("Layers", layers)
        self.mapping = nn.Sequential()
        for i in range(len(layers) - 2):
            self.mapping.add_module("fc_" + str(i), nn.Linear(layers[i], layers[i + 1]))
            self.mapping.add_module("bn_" + str(i), nn.BatchNorm1d(layers[i + 1]))
            self.mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # add last layer
        self.mapping.add_module("fc_" + str(len(layers) - 2), nn.Linear(layers[-2], layers[-1]))
        if len(layers) == 2:  # if only one layer, add batch normalization
            self.mapping.add_module("bn_" + str(len(layers) - 2), nn.BatchNorm1d(layers[-1]))
        #
        # # Attention
        # att_dim, semi_att_dim = 128, 128
        # self.use_att = use_att
        # if self.use_att:
        #     self.att_models = nn.ModuleList()
        #     for _ in range(nclass):
        #         att_models = nn.Sequential(
        #             nn.Linear(layers[-1], att_dim),
        #             nn.Tanh(),
        #             nn.Linear(att_dim, 1)
        #         )
        #         self.att_models.append(att_models)
        #
        # self.use_ss = use_ss  # whether to use semi-supervised mode
        # if self.use_ss:
        #     self.semi_att = nn.Sequential(
        #         nn.Linear(layers[-1], semi_att_dim),
        #         nn.Tanh(),
        #         nn.Linear(semi_att_dim, self.nclass)
        #     )

    def forward(self, input):
        # 数据的输入
        x, labels, idx_train, idx_val, idx_test = input  # x is N * L, where L is the time-series feature dimension
        # print("X:", x.shape)
        # print("idx_train",idx_train.shape)
        print("labels", labels.shape)

        if not self.use_muse:
            N = x.size(0)
        # Sequence Encoder
            # LSTM
            if self.use_lstm:
                x_lstm = self.lstm(x)[0]
                # print("x_lstm_shape", x_lstm.shape)
                x_lstm = x_lstm.mean(1)
                # print("x_lstm_mean", x_lstm.shape)
                x_lstm = x_lstm.view(N, -1)
                # print("lstm_out", x_lstm.shape)
                x_lstm = self.lstm_mapping(x_lstm)

                print("x_lstm_shpe", x_lstm.shape)

                # generate the class protocal with dimension C * D (nclass * dim)
                lstm_proto_list = []
                for i in range(self.nclass):
                    idx = (labels[idx_train].squeeze() == i).nonzero().squeeze(1)
                    if self.use_att:
                        # A = self.attention(x[idx_train][idx])  # N_k * 1
                        A = self.lstm_att_models[i](x_lstm[idx_train][idx])  # N_k * 1
                        A = torch.transpose(A, 1, 0)  # 1 * N_k
                        A = F.softmax(A, dim=1)  # softmax over N_k
                        # print("lstm_A", A.shape)
                        class_repr = torch.mm(A, x_lstm[idx_train][idx])  # 1 * L
                        class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
                        # print("lstm_class_repr", class_repr.shape)
                    else:  # if do not use attention, simply use the mean of training samples with the same labels.
                        class_repr = x[idx_train][idx].mean(0)  # L * 1
                    lstm_proto_list.append(class_repr.view(1, -1))
                lstm_x_proto = torch.cat(lstm_proto_list, dim=0)

            if self.use_cnn:
                # Covolutional Network
                # input ts: # N * C * L
                if self.use_rp:
                    for i in range(len(self.conv_1_models)):
                        # x_conv = x
                        x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                        x_conv = self.conv_bn_1(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_2(x_conv)
                        x_conv = self.conv_bn_2(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = self.conv_3(x_conv)
                        x_conv = self.conv_bn_3(x_conv)
                        x_conv = F.leaky_relu(x_conv)

                        x_conv = torch.mean(x_conv, 2)

                        if i == 0:
                            x_conv_sum = x_conv
                        else:
                            x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)

                    x_conv = x_conv_sum
                    print("cnn_out", x_conv.shape)
                else:
                    x_conv = x
                    x_conv = self.conv_1(x_conv)  # N * C * L
                    x_conv = self.conv_bn_1(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_2(x_conv)
                    x_conv = self.conv_bn_2(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_3(x_conv)
                    x_conv = self.conv_bn_3(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = x_conv.view(N, -1)

                # linear mapping to low-dimensional space
                cnn_x = self.cnn_mapping(x_conv)
                print("cnn_x_shpe", cnn_x.shape)

                # generate the class protocal with dimension C * D (nclass * dim)
                cnn_proto_list = []
                for i in range(self.nclass):
                    idx = (labels[idx_train].squeeze() == i).nonzero().squeeze(1)
                    if self.use_att:
                        # A = self.attention(x[idx_train][idx])  # N_k * 1
                        A = self.cnn_att_models[i](cnn_x[idx_train][idx])  # N_k * 1
                        A = torch.transpose(A, 1, 0)  # 1 * N_k
                        A = F.softmax(A, dim=1)  # softmax over N_k
                        # print("cnn_A", A.shape)
                        class_repr = torch.mm(A, cnn_x[idx_train][idx])  # 1 * L
                        class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
                        # print("cnn_class_repr", class_repr.shape)
                    else:  # if do not use attention, simply use the mean of training samples with the same labels.
                        class_repr = cnn_x[idx_train][idx].mean(0)  # L * 1
                    cnn_proto_list.append(class_repr.view(1, -1))
                cnn_x_proto = torch.cat(cnn_proto_list, dim=0)
            # print(x_proto)
            # dists = euclidean_dist(x, x_proto)
            # log_dists = F.log_softmax(-dists * 1e7, dim=1)
            if self.use_lstm and self.use_cnn:
                x = torch.cat([lstm_x_proto, cnn_x_proto], dim=1)
            elif self.use_lstm:
                x = lstm_x_proto
            elif self.use_cnn:
                x = cnn_x_proto

        # linear mapping to low-dimensional space
        x_proto = x
        print("x_proto", x_proto.shape)
        # x_proto = self.mapping(x_proto)
        # print("x_proto_shape", x_proto.shape)

        x = torch.cat([x_lstm, cnn_x], dim=1)
        print("embedding", x.shape)

        # x = self.mapping(x)

        # # generate the class protocal with dimension C * D (nclass * dim)
        # proto_list = []
        # for i in range(self.nclass):
        #     idx = (labels[idx_train].squeeze() == i).nonzero().squeeze(1)
        #     if self.use_att:
        #         # A = self.attention(x[idx_train][idx])  # N_k * 1
        #         A = self.att_models[i](x[idx_train][idx])  # N_k * 1
        #         A = torch.transpose(A, 1, 0)  # 1 * N_k
        #         A = F.softmax(A, dim=1)  # softmax over N_k
        #         print("A", A.shape)
        #         class_repr = torch.mm(A, x[idx_train][idx])  # 1 * L
        #         class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
        #         print("class_repr", class_repr.shape)
        #     else:  # if do not use attention, simply use the mean of training samples with the same labels.
        #         class_repr = x[idx_train][idx].mean(0)  # L * 1
        #     proto_list.append(class_repr.view(1, -1))
        # x_proto = torch.cat(proto_list, dim=0)

        # prototype distance
        proto_dists = euclidean_dist(x_proto, x_proto)
        num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
        proto_dist = torch.sum(proto_dists) / num_proto_pairs

        # print("proto_dist", proto_dist.shape)
        # print("proto_dist", proto_dist.size())

        if self.use_ss:
            semi_A = self.semi_att(x[idx_test])  # N_test * c
            semi_A = torch.transpose(semi_A, 1, 0)  # c * N_test
            semi_A = F.softmax(semi_A, dim=1)  # softmax over N_test
            x_proto_test = torch.mm(semi_A, x[idx_test])  # c * L
            x_proto = (x_proto + x_proto_test) / 2

            # solution 2
            # row_sum = 1 / torch.sum(-dists[idx_test,], dim=1)
            # row_sum = row_sum.unsqueeze(1).repeat(1, 2)
            # prob = torch.mul(-dists[idx_test,], row_sum)
            # x_proto_test = torch.transpose(torch.mm(torch.transpose(x[idx_test,], 0, 1), prob), 0, 1)

        dists = euclidean_dist(x, x_proto)

        # dump_embedding(x_proto, torch.from_numpy())
        dump_embedding(lstm_x_proto, x_lstm, labels, 'lstm_embeddings.txt')
        dump_embedding(cnn_x_proto, cnn_x, labels, 'cnn_embeddings.txt')
        dump_embedding(x_proto, x, labels, 'embeddings.txt')

        return -dists, proto_dist


