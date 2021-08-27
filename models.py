import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Downsample(nn.Module):
    r""" Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer('kernel', kernel[None, None, :].repeat((channels, 1, 1)))

    def forward(self, x):
        return F.conv1d(x, self.kernel, stride=self.stride, padding=self.padding, groups=x.shape[1])


class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:
          bn-relu-conv-bn-relu-conv
         /                         \
        x --------------------------(+)->
    """

    def __init__(
        self, in_channels, out_channels,
        kernel_size=3, stride=1, padding=1,
    ):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size, stride, padding,
                               bias=False, padding_mode='circular')
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size, stride, padding,
                               bias=False, padding_mode='circular')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))

        x = self.conv2(x)
        x = x + identity

        return x


class Resnet(nn.Module):
    r""" The general form of the architecture can be described as follows:

    x->[conv-[ResBlock]^m-bn-relu-down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-relu-conv-bn-relu-conv
           /                         \                      /                         \
    x->conv --------------------------(+)-bn-relu-down->conv --------------------------(+)-bn-relu-down-> ...

    """

    def __init__(self,
                 n_channels, outsize,
                 n_filters_list,
                 kernel_size_list,
                 n_resblocks_list,
                 resblock_kernel_size_list,
                 downfactor_list,
                 downorder_list,
                 drop1, drop2,
                 fc_size, is_cnnLSTM=False):
        super(Resnet, self).__init__()

        # Broadcast if single number provided instead of list
        if isinstance(kernel_size_list, int):
            kernel_size_list = [kernel_size_list] * len(downfactor_list)

        if isinstance(resblock_kernel_size_list, int):
            resblock_kernel_size_list = [resblock_kernel_size_list] * len(downfactor_list)

        if isinstance(n_resblocks_list, int):
            n_resblocks_list = [n_resblocks_list] * len(downfactor_list)

        cfg = zip(n_filters_list,
                  kernel_size_list,
                  n_resblocks_list,
                  resblock_kernel_size_list,
                  downfactor_list,
                  downorder_list)

        resnet = nn.Sequential()

        # Input channel dropout
        resnet.add_module('input_dropout', nn.Dropout2d(drop1))

        # Main layers
        in_channels = n_channels
        for i, layer_params in enumerate(cfg):
            out_channels, kernel_size, n_resblocks, resblock_kernel_size, downfactor, downorder = layer_params
            resnet.add_module(f'layer{i+1}', Resnet.make_layer(in_channels, out_channels,
                                                               kernel_size, n_resblocks, resblock_kernel_size,
                                                               downfactor, downorder))
            in_channels = out_channels

        if is_cnnLSTM is False:
            # Fully-connected layer
            resnet.add_module('fc', nn.Sequential(nn.Dropout2d(drop2),
                                                  nn.Conv1d(in_channels, fc_size, 1, 1, 0, bias=False),
                                                  nn.ReLU(True)))

            # Final linear layer
            resnet.add_module('final', nn.Conv1d(fc_size, outsize, 1, 1, 0, bias=False))

        self.resnet = resnet

    @staticmethod
    def make_layer(in_channels, out_channels,
                   kernel_size, n_resblocks, resblock_kernel_size,
                   downfactor, downorder):
        r""" Basic layer in Resnets:

        x->[conv-[ResBlock]^m-bn-relu-down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        assert kernel_size % 2, "Only odd number for conv_kernel_size supported"
        assert resblock_kernel_size % 2, "Only odd number for resblock_kernel_size supported"

        padding = int((kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [nn.Conv1d(in_channels, out_channels,
                             kernel_size, 1, padding,
                             bias=False, padding_mode='circular')]

        for _ in range(n_resblocks):
            modules.append(ResBlock(out_channels, out_channels,
                                    resblock_kernel_size, 1, resblock_padding))

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x):
        return self.resnet(x).reshape(x.shape[0], -1)


class CNNLSTM(nn.Module):
    def __init__(self, cnn_cfg, num_classes=2, lstm_layer=3, lstm_nn_size=1024,
                 model_device='cpu', dropout_p=0, bidrectional=False, batch_size=10):
        super(CNNLSTM, self).__init__()
        if bidrectional==True:
            fc_feature_size = lstm_nn_size*2
        else:
            fc_feature_size = lstm_nn_size
        self.fc_feature_size = fc_feature_size
        self.model_device = model_device
        self.lstm_layer = lstm_layer
        self.batch_size = batch_size
        self.lstm_nn_size = lstm_nn_size
        self.bidrectional = bidrectional

        self.feature_extractor = Resnet(
            cnn_cfg.model.n_channels,
            cnn_cfg.model.outsize,
            cnn_cfg.model.n_filters,
            cnn_cfg.model.kernel_size,
            cnn_cfg.model.n_resblocks,
            cnn_cfg.model.resblock_kernel_size,
            cnn_cfg.model.downfactor,
            cnn_cfg.model.downorder,
            cnn_cfg.model.drop1,
            cnn_cfg.model.drop2,
            cnn_cfg.model.fc_size,
            is_cnnLSTM=True)

        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_nn_size,
                            num_layers=lstm_layer, bidirectional=bidrectional)
        self.classifier = nn.Sequential(
            nn.Linear(fc_feature_size, fc_feature_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc_feature_size, fc_feature_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc_feature_size, num_classes)
        )

    def init_hidden(self, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        init_lstm_layer = self.lstm_layer
        if self.bidrectional:
            init_lstm_layer = self.lstm_layer*2
        hidden_a = torch.randn(init_lstm_layer, batch_size, self.lstm_nn_size, device=self.model_device)
        hidden_b = torch.randn(init_lstm_layer, batch_size, self.lstm_nn_size, device=self.model_device)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, x, seq_lengths):
        # x dim: batch_size x C x F_1
        # we will need to do the packing of the sequence dynamically for each batch of input
        # 1. feature extractor
        x = self.feature_extractor(x)  # x dim: total_epoch_num * feature size
        if x.size()[-1] == 1:
            x = torch.squeeze(x, 2) # force the last dim to be of feature size
        feature_size = x.size()[-1]

        # 2. lstm
        seq_tensor = torch.zeros(len(seq_lengths), seq_lengths.max(),
                                feature_size, dtype=torch.float,
                                device=self.model_device)
        start_idx = 0

        for i in range(len(seq_lengths)):

            current_len = seq_lengths[i]
            current_series = x[start_idx:start_idx+current_len, :]  # num_time_step x feature_size
            current_series = current_series.view(1, current_series.size()[0], -1)

            seq_tensor[i, :current_len, :] = current_series
            start_idx += current_len

        seq_lengths_ordered, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        packed_input = pack_padded_sequence(seq_tensor, seq_lengths_ordered.cpu().numpy(), batch_first=True)

        # x dim for lstm: #  batch_size_rnn x Sequence_length x F_2
        # uncomment for random init state
        # hidden = self.init_hidden(len(seq_lengths))
        packed_output, _ = self.lstm(packed_input)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # reverse back to the original order
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = output[unperm_idx]

        # reverse back to the originaly shape
        # total_epoch_num * fc_feature_size
        fc_tensor = torch.zeros(seq_lengths.sum(), self.fc_feature_size,
                                dtype=torch.float, device=self.model_device)

        start_idx = 0
        for i in range(len(seq_lengths)):
            current_len = seq_lengths[i]
            current_series = lstm_output[i, :current_len, :]  # num_time_step x feature_size
            current_series = current_series.view(current_len, -1)
            fc_tensor[start_idx:start_idx + current_len, :] = current_series
            start_idx += current_len

        # 3. linear readout
        x = self.classifier(fc_tensor)
        return x
