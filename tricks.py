import numpy as np
from tqdm import tqdm
import random
import os
import torch


def fit_test_th(x, y):
    p = []
    for idx in tqdm(range(len(y))):
        _y = y[idx]
        _x = x[:, idx]
        min_error = np.inf
        min_p = 0
        for _p in np.linspace(0, 1, 10000):
            error = np.abs((_x > _p).mean() - _y)
            if error < min_error:
                min_error = error
                min_p = _p
            elif error == min_error and (np.abs(_p - 0.5) < np.abs(min_p - 0.5)):
                min_error = error
                min_p = _p
        p.append(min_p)
    p = np.array(p)
    return p


def seed_everything(seed=7):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)