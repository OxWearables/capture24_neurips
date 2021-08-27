import sys
from urllib import request
from .utils import onehot
from scipy.interpolate import interp1d

import random
import numpy as np
import torch
import math
from torch.utils.data import Dataset

from transforms3d.axangles import axangle2mat  # for rotation

sys.path.append("../semi-supervised")
n_labels = 10
cuda = torch.cuda.is_available()


def get_seq_lens(pid_list):
    lens = np.where(pid_list[:-1] != pid_list[1:])[0]
    lens = np.concatenate((lens, [len(pid_list) - 1]))
    seq_lengths = []
    pre_len = -1
    for my_len in lens:
        seq_lengths.append(my_len - pre_len)
        pre_len = my_len

    return torch.LongTensor(seq_lengths)


def train_test_split(X, y, group, num_test, context_df=None, time_df=None):
    """
    Get num_test subjects out from the X to be saved as test
    The rest will be treated as train
    """
    subject_list = np.unique(group)
    test_subject = np.random.choice(subject_list, num_test, replace=False)
    print(test_subject)
    test_idx = np.isin(group, test_subject)
    train_idx = ~test_idx

    X_test = X[test_idx]
    y_test = y[test_idx]
    group_test = group[test_idx]

    X_trian = X[train_idx]
    y_train = y[train_idx]
    group_train = group[train_idx]

    if context_df is not None:
        context_test = context_df[test_idx]
        context_train = context_df[train_idx]
    else:
        context_train = None
        context_test = None

    if time_df is None:
        return X_trian, X_test, y_train, y_test, group_train, group_test, context_train, context_test
    else:
        time_test = time_df[test_idx]
        time_train = time_df[train_idx]
        return X_trian, X_test, y_train, y_test, group_train, group_test, \
               time_train, time_test, context_train, context_test


# Taken from https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)


class Permutation_TimeSeries(object):
    """
    Rotation along an axis
    """

    def __call__(self, sample):
        # print("sampel shape")
        # print(sample.shape)
        # TIME_STEP * 3 * FEATURE_SIZE
        sample = np.swapaxes(sample, 1, 2)
        # MIN one segment
        sample = np.array([DA_Permutation(xi, nPerm=max(math.ceil(np.random.normal(2, 5)), 1)) for xi in sample])

        sample = np.swapaxes(sample, 1, 2)
        sample = torch.tensor(sample)
        return sample


class RotationAxisTimeSeries(object):
    """
    Every sample belongs to one subject
    Rotation along an axis
    """

    def __call__(self, sample):
        # print("sampel shape")
        # print(sample.shape)
        # TIME_STEP * 3 * FEATURE_SIZE
        axis = np.random.uniform(low=-1, high=1, size=sample.shape[1])
        angle = np.random.uniform(low=-np.pi, high=np.pi)

        sample = np.swapaxes(sample, 1, 2)
        sample = np.matmul(sample, axangle2mat(axis, angle))

        sample = np.swapaxes(sample, 1, 2)
        # sample = torch.tensor(sample)
        return sample


def resize(X, length, axis=1):
    ''' Resize the temporal length using linear interpolation.
    X must be of shape (N,M,C) (channels last) or (N,C,M) (channels first),
    where N is the batch size, M is the temporal length, and C is the number
    of channels.
    If X is channels-last, use axis=1 (default).
    If X is channels-first, use axis=2.
    '''

    length_orig = X.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    X = interp1d(t_orig, X, kind='linear', axis=axis, assume_sorted=True)(t_new)

    return X


class RandomSwitchAxisTimeSeries(object):
    """
    Randomly switch the three axises for the raw files
    """

    def __call__(self, sample):
        # print(sample.shape)
        # TIME_STEP * 3 * FEATURE_SIZE
        x = sample[:, 0, :]
        y = sample[:, 1, :]
        z = sample[:, 2, :]

        choice = random.randint(1, 6)
        if choice == 1:
            sample = torch.stack([x, y, z], dim=1)
        elif choice == 2:
            sample = torch.stack([x, z, y], dim=1)
        elif choice == 3:
            sample = torch.stack([y, x, z], dim=1)
        elif choice == 4:
            sample = torch.stack([y, z, x], dim=1)
        elif choice == 5:
            sample = torch.stack([z, x, y], dim=1)
        elif choice == 6:
            sample = torch.stack([z, y, x], dim=1)
        # print(sample.shape)
        return sample


class RandomSwitchAxis(object):
    """
    Randomly switch the three axises for the raw files
    Input size: 3 * FEATURE_SIZE
    """

    def __call__(self, sample):
        # print(sample.shape)
        # 3 * FEATURE
        x = sample[0, :]
        y = sample[1, :]
        z = sample[2, :]

        choice = random.randint(1, 6)

        if choice == 1:
            sample = torch.stack([x, y, z], dim=0)
        elif choice == 2:
            sample = torch.stack([x, z, y], dim=0)
        elif choice == 3:
            sample = torch.stack([y, x, z], dim=0)
        elif choice == 4:
            sample = torch.stack([y, z, x], dim=0)
        elif choice == 5:
            sample = torch.stack([z, x, y], dim=0)
        elif choice == 6:
            sample = torch.stack([z, y, x], dim=0)
        return sample


class RotationAxis(object):
    """
    Rotation along an axis
    """

    def __call__(self, sample):
        # 3 * FEATURE_SIZE
        sample = np.swapaxes(sample, 0, 1)
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        axis = np.random.uniform(low=-1, high=1, size=sample.shape[1])
        sample = np.matmul(sample, axangle2mat(axis, angle))
        sample = np.swapaxes(sample, 0, 1)
        return sample
