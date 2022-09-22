import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *
from random import sample
from tqdm import trange
from scipy.stats import sem


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    return F.dropout(x, drop_prob) * (1 - drop_prob)


def save_checkpoint(fname, **kwargs):
    checkpoint = {}
    for key, value in kwargs.items():
        checkpoint[key] = value

    torch.save(checkpoint, fname)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True