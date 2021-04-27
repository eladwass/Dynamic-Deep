import numpy as np
import torch
from scipy.stats import entropy
from wfdb import processing
cls = np


def calc_PRD_matrices(original, reconstructed):
    return torch.mean(100 * torch.sqrt(torch.sum((original - reconstructed) ** 2, dim=1) / torch.sum(original ** 2, dim = 1)))

def calc_PRD(original, reconstructed):
    return np.sqrt((np.linalg.norm(original-reconstructed)**2/np.linalg.norm(original)**2))*100


def calc_ECG_annotation(annotation_GT, reconstructed, fs=360):
    qrs_inds = processing.xqrs_detect(sig=reconstructed, fs=fs,verbose=True)
    # Compare detected qrs complexes to reference annotation.
    # Note, first sample in 100.atr is not a qrs.ֶ
    comparitor = processing.compare_annotations(ref_sample=annotation_GT,
                                                test_sample=qrs_inds,
                                                window_width=int(0.1 * fs),
                                                signal=reconstructed)

    F_mesure = 2 * (comparitor.positive_predictivity * comparitor.sensitivity) / (
                comparitor.positive_predictivity + comparitor.sensitivity)
    return [len(annotation_GT), comparitor.n_test, comparitor.tp, comparitor.fp, comparitor.fn,
           comparitor.positive_predictivity, comparitor.sensitivity, F_mesure]



def calc_loss_percentage_single(original, reconstructed):
    # for each sample in window_size calc the percentage loss and then avarage it
    return (100 * cls.mean(cls.abs((reconstructed - original) / (original+cls.random.rand(original.shape[0])*0.001))))

def calc_loss_percentage(original, reconstructed):
    # for each sample in window_size calc the percentage loss and then avarage it
    return (100 * torch.mean(torch.abs((reconstructed - original) / (original+torch.rand(original.shape[1])*0.001)))).numpy()


def calc_max_loss(original, reconstructed):
    return (100 * torch.max(torch.abs((reconstructed - original) / (original+torch.rand(original.shape[1])*0.001)))).numpy()


def calc_min_loss(original, reconstructed):
    return (100 * torch.min(torch.abs((reconstructed - original) / (original+torch.rand(original.shape[1])*0.001)))).numpy()


def calc_entropy(vector):
    value, counts = np.unique(vector, return_counts=True)
    return entropy(counts, base=None)


def calc_STD(vector):
    return np.std(vector)


def BatchNrom_constant(local_batch):
    local_batch = (local_batch - torch.mean(local_batch,dim=1)[:,None]) / torch.sqrt(torch.var(local_batch,dim=1))[:,None]
    return local_batch


#        (b-a)(x - min)
# f(x) = --------------  + a
#           max - min
def normelize_data(X, a, b, min_val=-1, max_val=-1, cls=np):
    '''
    normelize data range to be -> [0:1]
    '''
    if min_val == -1:
        min_val = cls.min(X)
    if max_val == -1:
        max_val = cls.max(X)

    X = X - min_val
    # print ("max value of all samples:", max_value)
    X = (X / (max_val - min_val)) * (b - a) + a
    return X


def STD_cleaning(local_batch,threshold,local_labels):
    idxs = torch.nonzero(torch.std(local_batch,dim=1)<threshold)
    return local_batch[idxs].squeeze(dim=1),local_labels[idxs].squeeze(dim=1)
