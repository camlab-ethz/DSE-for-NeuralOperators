

import torch
from functools import reduce
import operator



# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c


def percentage_difference(truth, test):
    difference = torch.mean(torch.abs(truth - test))/torch.mean(torch.abs(truth)) * 100
    return difference.item()