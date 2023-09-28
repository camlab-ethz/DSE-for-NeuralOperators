

import torch
from functools import reduce
import operator
import scipy
import h5py
import numpy as np


def count_params(model):
    """
    Print the number of parameters
    """
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c


def percentage_difference(truth, test):
    """
    Compute relative errors
    """
    difference = torch.mean(torch.abs(truth - test))/torch.mean(torch.abs(truth)) * 100
    return difference.item()


def normalize(data):
    """
    Normalizes all data to [0, 1] range.
    """
    data_min = torch.min(data)
    data_max = torch.max(data)
    
    new_data = (data - data_min) / (data_max - data_min)
    return new_data


class MatReader(object):
    """
    Reading data from .mat files
    """
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

