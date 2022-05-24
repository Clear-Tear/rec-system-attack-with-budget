import numpy as np
import pandas as pd


def data_loader(path):
    price_data = pd.read_csv(path, sep=' ', header=None, names=['user', 'price'], usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    return price_data
