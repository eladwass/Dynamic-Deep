import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils import data
import os
from os.path import join

SAMPLE_SIZE = 127
# from pathlib import Path


def train_test_dataset(folder_path, params,Dataset):
    # TODO - moduleName : change the Dataset module to a dictionary {'ModuleName':module from import}
    '''
    :param folder_path:
    :param params: dictionary with batch_size,shuffle, num_workers
    :param Dataset: the data loader
    :return:
    '''
    # Datasets

    print(len(folder_path))

    train, val = train_test_split(folder_path, test_size=0.4, random_state=11)
    print('train:', len(train), 'test:', len(val))
    partition = {'train': train, 'validation': val}
    labels = []  # Labels #if it is AE not need for labels

    # Generators
    training_set = Dataset(partition['train'], labels, 'float32', ((0, 1)))
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(partition['validation'], labels, 'float32', ((0, 1)))
    validation_generator = data.DataLoader(validation_set, **params)
    # partition
    return training_generator, validation_generator


def split_data(data_arr, window_size = SAMPLE_SIZE):
    num_blocks = len(data_arr) // window_size
    throw = (len(data_arr) % window_size)
    if throw != 0:
        X = np.array(np.split(data_arr[0:-throw], num_blocks))
    else:
        X = np.array(np.split(data_arr, num_blocks))

    return X

def dir_list(rootdir):
    list = []
    for root, subdirs, files in os.walk(rootdir):
        list+=[join(root, file) for file in files]
    return list
