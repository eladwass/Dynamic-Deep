import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile,join
import torch
import wfdb
from torch.utils import data

bit_per_sample = 11  # not used
freq = 360  # samples in sec - not used
SAMPLE_SIZE = 127
default_path_company = r'..\..\Data\MIT_BIH_CSV'
NUM_of_windows = {
    4096: 158,
    2048: 316
}

class Dataset(data.Dataset):
    '''
    Characterizes a dataset for PyTorch
    '''

    def __init__(self, list_IDs, labels, typecast='float32',Normalize_scale=(0,1),WINDOW_SIZE=4096):
        'Initialization'
        self.labels = labels

        ranges = [(x*WINDOW_SIZE, (x+1)*WINDOW_SIZE) for x in range(NUM_of_windows[WINDOW_SIZE])]
        self.final_dataset = []
        for file_name in list_IDs:
            single_list = [(file_name,range) for range in ranges]
            self.final_dataset += single_list
        print('MIT_BIH_loader initialized')
        self.norm_scale = Normalize_scale
        self.WINDOW_SIZE = WINDOW_SIZE

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.final_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_name, range = self.final_dataset[index]
        sempfrom,sempto = range
        # Load data and get label
        X, y = load_data(file_name, sempfrom, sempto)

        X = torch.from_numpy(np.array(X)).float()
        if type(self.norm_scale) == type((0, 1)):
            X = (X - torch.min(X)) / (torch.max(X) - torch.min(X)) * (self.norm_scale[1] - self.norm_scale[0]) + \
                self.norm_scale[0]
        y = torch.from_numpy(np.array(y)).int()

        return X, y, file_name,sempfrom


def load_data(data__dir_path, offset=0, sampto='end'):
    only_files = [data__dir_path]
    if not isfile(data__dir_path):
        only_files = [join(data__dir_path, f) for f in listdir(data__dir_path) if isfile(join(data__dir_path, f))]
    data_arr = np.array([])
    anotation_arr = np.array([])
    for idx, file_path in enumerate(only_files):
        file_path = file_path.split('.')[0]
        signals, _ = wfdb.rdsamp(file_path, sampfrom=offset, sampto=sampto)

        data_file = signals[:,0]  # ['\'V5\''] another option
        data_arr = np.concatenate((data_arr, data_file))

        try:
            annotation_file = wfdb.rdann(file_path, 'atr', sampfrom=offset, sampto=sampto).sample - offset
            anotation_arr = np.concatenate((anotation_arr, annotation_file.astype('int32')))
        except Exception as e:
            print(str(e))

        if idx % 10 == 0 and idx > 0:
            print((0.0+idx)/len(only_files))

    return data_arr, anotation_arr
