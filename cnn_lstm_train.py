import os
import numpy as np
import pandas as pd
import time
import random
import collections
import hydra
from omegaconf import DictConfig, OmegaConf
import copy

# Model utils
from utilities.utils import onehot, compute_scores, gradient_clip_lstm
from utilities.datautils import train_test_split, RandomSwitchAxis, RandomSwitchAxisTimeSeries, \
    RotationAxisTimeSeries, Permutation_TimeSeries, resize
from utilities.pytorchtools import EarlyStopping

# VGG-variant for sliding window
from models import CNNLSTM
from utilities.weight_init import weight_init

# Data utils
from utilities.data_loader import cnnLSTMDataset, SubjectDataset
from sklearn import preprocessing
from utils import metrics_report

# Torch
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from torchvision import transforms

# Plotting
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

cuda = torch.cuda.is_available()
now = datetime.now()

################################
#
#
#       helper functions
#
#
################################
def report_model_performance(model, my_train_loader, my_test_loader, my_writer, my_device, loss_fn, epoch):
    epoch_train_scores, train_loss = evaluate_model(model, my_train_loader, my_device, loss_fn)
    epoch_val_scores, val_loss = evaluate_model(model, my_test_loader, my_device, loss_fn)

    print(epoch_train_scores)
    print(epoch_val_scores)
    my_writer.add_scalar('Train/loss', train_loss, epoch)
    my_writer.add_scalar('Val/loss', val_loss, epoch)

    my_writer.add_scalar('Train/kappa', epoch_train_scores['kappa'], epoch)
    my_writer.add_scalar('Val/kappa', epoch_val_scores['kappa'], epoch)

    my_writer.add_scalar('Train/accuracy', epoch_train_scores['accuracy'], epoch)
    my_writer.add_scalar('Val/accuracy', epoch_val_scores['accuracy'], epoch)

    my_writer.add_scalar('Train/b_accuracy', epoch_train_scores['balanced_accuracy'], epoch)
    my_writer.add_scalar('Val/b_accuracy', epoch_val_scores['balanced_accuracy'], epoch)

    my_writer.add_scalar('Train/f1', epoch_train_scores['f1'], epoch)
    my_writer.add_scalar('Val/f1', epoch_val_scores['f1'], epoch)

    my_writer.add_scalar('Train/phi', epoch_train_scores['phi'], epoch)
    my_writer.add_scalar('Val/phi', epoch_val_scores['phi'], epoch)
    return epoch_train_scores, epoch_val_scores, train_loss, val_loss


def evaluate_model(model, my_data_loader, my_device, loss_fn):
    model.eval()
    test_y_pred = []
    test_y_real = []

    current_loss = 0

    for i, (subject_X, subject_y, subject_pid) in enumerate(my_data_loader):
        subject_dataset = SubjectDataset(subject_X, subject_y, subject_pid)
        subject_loader = DataLoader(subject_dataset, batch_size=1, shuffle=False)

        for j, (my_X, my_y, my_pid) in enumerate(subject_loader):
            my_X, my_y = Variable(my_X), Variable(my_y)
            my_X = my_X.to(my_device, dtype=torch.float)
            my_y = my_y.to(my_device, dtype=torch.float)

            my_X = torch.squeeze(my_X)
            my_y = torch.squeeze(my_y)
            my_pid = torch.squeeze(my_pid)

            seq_lengths = get_seq_lens_hour_long(my_pid)

            with torch.no_grad():
                logits = model(my_X, seq_lengths)
                target_y = torch.argmax(my_y, dim=1).view(-1)
                loss = loss_fn(logits, target_y)
                current_loss += loss.item()

                _, pred_idx = torch.max(logits, 1)
                _, lab_idx = torch.max(my_y, 1)

                test_y_pred.extend(torch.max(logits, 1)[1].cpu().detach().numpy())
                test_y_real.extend(torch.max(my_y, 1)[1].cpu().detach().numpy())

    current_loss = current_loss / len(my_data_loader)
    test_y_pred = np.stack(test_y_pred)
    test_y_real = np.array(test_y_real)

    metrics_report(test_y_real, test_y_pred)
    return compute_scores(test_y_real, test_y_pred), current_loss


def set_seed(my_seed=0):
    random_seed = my_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if cuda:
        torch.cuda.manual_seed_all(random_seed)


def cnn_lstm_collate(batch):
    num_class = 6
    data = [item[0] for item in batch]
    target = [torch.cat(item[1]) for item in batch]
    pid = [item[2] for item in batch]

    pid = np.concatenate(pid)
    pid = torch.Tensor(pid)

    data = torch.cat(data)

    target = torch.cat(target)
    target = target.view(-1, num_class).long()
    target = torch.LongTensor(target)

    return [data, target, pid]


def get_seq_lens(pid_list):
    lens = np.where(pid_list[:-1] != pid_list[1:])[0]
    lens = np.concatenate((lens, [len(pid_list) - 1]))
    seq_lengths = []
    pre_len = -1
    for my_len in lens:
        seq_lengths.append(my_len - pre_len)
        pre_len = my_len

    return torch.LongTensor(seq_lengths)


def split_len(seq_len, max_seq_len):
    # split a sequence len into smaller chunk
    remainder = seq_len % max_seq_len
    full_sequence_count = int(seq_len / max_seq_len)
    lengths = []
    if full_sequence_count == 0:
        lengths.append(remainder)
    else:
        lengths = full_sequence_count * [max_seq_len]
        if remainder != 0:
            lengths.append(remainder)
    return lengths


def get_seq_lens_hour_long(pid_list,
                           sequence_len=3,
                           epoch_len=10):
    # sequence_len: hour
    # epoch_len: sec
    # split time series into lengths of certain hours
    seconds_per_hour = 60*60
    max_seq_len = sequence_len*seconds_per_hour/epoch_len

    # 1. get the sequence length for each subject
    if len(np.unique(pid_list)) == 1:
        seq_lengths = [len(pid_list)]
    else:
        lens = np.where(pid_list[:-1] != pid_list[1:])[0]
        lens = np.concatenate((lens, [len(pid_list) - 1]))
        seq_lengths = []
        pre_len = -1
        for my_len in lens:
            seq_lengths.append(my_len - pre_len)
            pre_len = my_len

    # 2. split each sequence into the shorter interval
    final_seq_lengths = []
    for my_len in seq_lengths:
        final_seq_lengths = final_seq_lengths + split_len(my_len, max_seq_len)

    return torch.LongTensor(final_seq_lengths)


def load_weights(weight_path, model, my_device, name_start_idx=1):
    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(pretrained_dict)  # v2 has the right para names
    for key in pretrained_dict:
        para_names = key.split('.')
        para_names = ['feature_extractor'] + para_names[name_start_idx:]
        new_key = '.'.join(para_names)
        pretrained_dict_v2[new_key] = pretrained_dict_v2.pop(key)

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys such as the final linear layers
    #    we don't want linear layer weights either
    pretrained_dict = {k: v for k, v in pretrained_dict_v2.items() if k in model_dict
                       and k.split('.')[0] != 'classifier'}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print("%d Weights loaded" % len(pretrained_dict))


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        m.eval()


def freeze_weights(model):
    i = 0
    # Set Batch_norm running stats to be frozen
    # or it will lead to bad results http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
    for name, param in model.named_parameters():
        print(name)
        if name.split('.')[0] == 'feature_extractor':
            param.requires_grad = False
            i += 1
    print("Weights being frozen: %d" % i)
    model.apply(set_bn_eval)


@hydra.main(config_path="conf", config_name="config_cnnlstm")
def main(cfg):
    set_seed()
    print(OmegaConf.to_yaml(cfg))

    ####################
    #   Setting macros
    ###################
    num_epochs = cfg.experiment.n_epochs
    lr = cfg.model.learning_rate  # learning rate in SGD
    batch_size = cfg.data.batch_size
    GPU = cfg.gpu
    multi_gpu = cfg.multi_gpu
    gpu_ids = cfg.gpu_ids
    useAugment = cfg.augment

    main_log_dir = '/home/cxx579/capture24_neurips/logs'
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    log_dir = os.path.join(main_log_dir, dt_string + '_weighted_cost' + str(cfg.optim.weighted_cost))
    model_path = os.path.join(main_log_dir, 'models', dt_string + '_weighted_cost' + str(cfg.optim.weighted_cost) + '.mdl')
    gradient_clip_val = cfg.model.graident_clip_val
    dropout_p = cfg.model.dropout_p
    is_biLSTM = cfg.model.bi_lstm
    lstm_nn_size = cfg.model.lstm_nn_size
    lstm_layer = cfg.model.lstm_layer

    print("Model name: %s" % cfg.model.name)
    print("Learning rate: %f" % lr)
    print("Number of epoches: %d" % num_epochs)
    print("GPU usage: %d" % GPU)
    print("Batch size: %d" % batch_size)
    print("Tensor log dir: %s" % log_dir)
    print("Use augmentation: %r" % useAugment)
    print("Model path to store: %s" % model_path)

    writer = SummaryWriter(log_dir)
    ####################
    #   Load data
    ###################
    start = time.time()
    print("Loading X raw...")
    X = np.load(cfg.data.X_path)
    Y = np.load(cfg.data.Y_path)
    pid = np.load(cfg.data.pid_path)

    # encoding
    le = preprocessing.LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)
    pid_le = preprocessing.LabelEncoder()
    pid_le.fit(pid)

    end = time.time()
    print("Loading completed. Used: %d sec" % (end - start))
    # feat_size = 300 # downsampled to 30Hz
    # X = resize(X, feat_size)
    X = np.swapaxes(X, 1, 2) # change to the format of N x C x FEATURE_SIZE
    print(X.shape)

    num_classes = 6

    num_subjects = len(np.unique(pid))
    print("#unique subjects %d" % num_subjects)

    # deriv/test split: P001-P100 for derivation, the rest for testing
    whr_deriv = np.isin(pid, [f"P{i:03d}" for i in range(1, 101)])
    X_train, y_train, pid_train = X[whr_deriv], Y[whr_deriv], pid[whr_deriv]
    X_test, y_test, pid_test = X[~whr_deriv], Y[~whr_deriv], pid[~whr_deriv]

    # further split deriv into train/val
    # val_pid = np.random.choice(np.unique(pid_train),
    #                           size=cfg.data.val_size,
    #                           replace=False)
    val_pid =['P001', 'P005', 'P011', 'P013', 'P019', 'P023', 'P031', 'P032',
              'P034', 'P040', 'P045', 'P046', 'P054', 'P071', 'P074', 'P077',
              'P078', 'P081', 'P084', 'P091']
    whr_val = np.isin(pid_train, val_pid)

    X_val, y_val, pid_val = X_train[whr_val], y_train[whr_val], pid_train[whr_val]
    X_train, y_train, pid_train = X_train[~whr_val], y_train[~whr_val], pid_train[~whr_val]

    pid_train = pid_le.transform(pid_train)
    pid_val = pid_le.transform(pid_val)
    pid_test = pid_le.transform(pid_test)
    print("X_train shape: " + str(X_train.shape))
    print("y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("y_test shape: " + str(y_test.shape))
    print("Train %d val %d  test %d " % (len(np.unique(pid_train)),
                                         len(np.unique(pid_val)),
                                         len(np.unique(pid_test))))
    # ####################
    # #   Model construction
    # ###################
    num_workers = 5
    if GPU != -1:
        my_device = 'cuda:' + str(GPU)
    else:
        my_device = 'cpu'

    # when temporal depedency is not consdiered, we should shuffle everything
    # make sure augment is cnnlstm when using cnnlstm
    myTransform = None
    if useAugment=='cnn_lstm':
        myTransform = transforms.Compose([
            RandomSwitchAxisTimeSeries(),
            RotationAxisTimeSeries(),
            Permutation_TimeSeries()
        ])
    train_dataset = cnnLSTMDataset(X_train,
                                   y=y_train,
                                   pid=pid_train,
                                   transform=myTransform,
                                   target_transform=onehot(num_classes))
    test_dataset = cnnLSTMDataset(X_test,
                                   y=y_test,
                                   pid=pid_test,
                                   transform=myTransform,
                                   target_transform=onehot(num_classes))
    val_dataset = cnnLSTMDataset(X_val,
                                   y=y_val,
                                   pid=pid_val,
                                   transform=myTransform,
                                   target_transform=onehot(num_classes))

    train_loader = DataLoader(train_dataset,
                    batch_size=batch_size,
                    collate_fn=cnn_lstm_collate,
                    shuffle=True,
                    num_workers=num_workers)
    test_loader = DataLoader(test_dataset,
                        batch_size=batch_size,
                        collate_fn=cnn_lstm_collate,
                        shuffle=False,
                        num_workers=num_workers)
    val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        collate_fn=cnn_lstm_collate,
                        num_workers=num_workers)

    model = CNNLSTM(cfg, num_classes=num_classes, model_device=my_device, lstm_nn_size=lstm_nn_size,
                    dropout_p=dropout_p, bidrectional=is_biLSTM, lstm_layer=lstm_layer)
    weight_init(model)
    print(model)
    if multi_gpu:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(my_device, dtype=torch.float)

    if cfg.eval:
        model.load_state_dict(torch.load(cfg.trained_path, map_location=my_device))


    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num of paras %d " % pytorch_total_params)

    # ## Weighted cost function, inverse of the ratio
    num_samples = len(y_train)
    if cfg.optim.weighted_cost:
        counter = collections.Counter(y_train)
        weights = [0] * num_classes
        for idx in counter.keys():
            weights[idx] = 1.0 / (counter[idx] / num_samples)
        weights = torch.FloatTensor(weights).to(my_device)
        print("Weight tensor: ")
        print(weights.cpu().detach())
        loss_fn = nn.CrossEntropyLoss(weight=weights)

    else:
        loss_fn = nn.CrossEntropyLoss()

    if cfg.model.pre_trained and cfg.model.freeze_all:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, amsgrad=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)

    total_step = len(train_loader)
    test_scores = []
    train_scores = []

    if cfg.eval:
        test_scores, test_loss = evaluate_model(model, test_loader,
                                                my_device, loss_fn)
        print(test_scores)
        return


    print("Start training")
    early_stopping = EarlyStopping(patience=cfg.model.patience,
                                   path=model_path,
                                   verbose=True)

    for epoch in range(num_epochs):
        model.train()
        current_loss = 0

        for i, (subject_X, subject_y, subject_pid) in enumerate(train_loader):
            subject_dataset = SubjectDataset(subject_X, subject_y, subject_pid)
            subject_loader = DataLoader(subject_dataset, batch_size=1, shuffle=False)

            for j, (my_X, my_y, my_pid) in enumerate(subject_loader):
                my_X, my_y = Variable(my_X), Variable(my_y)
                my_X = my_X.to(my_device, dtype=torch.float)
                my_y = my_y.to(my_device, dtype=torch.float)

                my_X = torch.squeeze(my_X)
                my_y = torch.squeeze(my_y)
                my_pid = torch.squeeze(my_pid)

                seq_lengths = get_seq_lens_hour_long(my_pid)

                # Forward pass
                outputs = model(my_X, seq_lengths)

                target_y = torch.argmax(my_y, dim=1).view(-1)
                loss = loss_fn(outputs, target_y)
                current_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                gradient_clip_lstm(model, gradient_clip_val)
                loss.backward()
                optimizer.step()

                if (i + 1) % 5 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Logging
        epoch_train_scores, epoch_val_scores, train_loss, val_loss = report_model_performance(model,
                                                                                                train_loader,
                                                                                                val_loader,
                                                                                                writer,
                                                                                                my_device,
                                                                                                loss_fn,
                                                                                                epoch)
        train_scores.append(epoch_train_scores)
        test_scores.append(epoch_val_scores)
        early_stopping(val_loss, model)

        # if val_loss < best_loss:
        #     print("Saving model with val loss %.4f -> %.4f" % (best_loss, val_loss))
        #     best_loss = val_loss
        #     torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
