import json
import pickle

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import torch

import yaml


class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.pts = 0

    def update(self, val, n=1):
        self.sum += val
        self.pts += n

    @property
    def value(self):
        return self.sum / self.pts


def get_experiment_name(configs, abbrevs=None):
    if abbrevs is None:
        abbrevs = {}

    for key, value in configs.items():
        if isinstance(value, dict):
            get_experiment_name(value, abbrevs)
        else:
            i = 1
            while i <= len(key):
                if key[:i] not in abbrevs:
                    abbrevs[key[:i]] = str(value).replace(" ", "").replace(",", "_").replace("[", "").replace("]", "")
                    break
                i += 1

                if i == len(key) + 1:
                    raise ValueError("Could not find a suitable abbreviation for key: {}".format(key))

    return abbrevs


def save_object(obj, filename):
    with open(filename, 'wb') as out_file:
        pickle.dump(obj, out_file, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as inp_file:  # Overwrites any existing file.
        out = pickle.load(inp_file)
    return out


def save_json(dct, path):
    with open(path, 'w') as f:
        json.dump(dct, f)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_yaml(dct, path):
    with open(path, 'w') as f:
        yaml.dump(dct, f)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f)


def apply_padding(source, target, mask=None, padding_value=-100):
    """ Apply padding to source and target sequences.

    Args:
        source: [[T1, D], [T2, D], ...]
        target: [[T1], [T2], ...]   \in [0, 1, 2, 3, 4]
        mask: [[T1], [T2], ...]     \in [0, 1]

    Returns:
        source: [B, T, D]
        target: [B, T]
        batch_mask: [B, T]
    """

    # Apply mask
    if mask is not None:
        source = [s[m == 1] for s, m in zip(source, mask)]  # [[T1', D], [T2', D], ...]
        target = [t[m == 1] for t, m in zip(target, mask)]  # [[T1'], [T2'], ...]

    # Apply padding
    source = torch.nn.utils.rnn.pad_sequence(
        source, batch_first=True, padding_value=padding_value
     )

    target = torch.nn.utils.rnn.pad_sequence(
        target, batch_first=True, padding_value=padding_value
    )

    batch_mask = (target != padding_value) * 1

    return source, target, batch_mask