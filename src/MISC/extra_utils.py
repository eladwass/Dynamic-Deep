import numpy as np
import torch


def read_file_with_jumps(path, offset=4):
    with open(path, 'rb') as my_file:
        data = np.frombuffer(my_file.read(), dtype='double')
        len_res = data[0]

        jumps = (len(data)-1) / len_res
        end = len_res*jumps
        loc = np.linspace(offset, len_res*jumps-(jumps-offset), len_res).astype('int')
        arr = np.array(data[loc])

    return arr


def reshapeECG_data(X, Y, dimy, device):
    local_batch, local_labels = X.to(device), Y.to(device)
    #     local_batch,_ = Data_utils.remove_freqeuncy(frequency_remove,local_batch)
    # original size is -> params['batch_size'][WINDOWSIZE*64]
    prev_shape = local_batch.shape[0]
    local_batch = local_batch.view(-1, dimy)
    labels_toreturn = []
    #     print(np.linspace(0,prev_shape-dimy,prev_shape//dimy))
    for i in np.linspace(0, prev_shape - dimy, prev_shape // dimy).astype('int32'):
        label_window = local_labels[local_labels < ((i + 1) * dimy - 1)]  # [local_labels>(i)*dimy]
        labels_toreturn.append(label_window)
    return local_batch, labels_toreturn


def BatchNrom_constant(local_batch):
    local_batch = (local_batch - torch.mean(local_batch, dim=1)[:,None]) / torch.sqrt(torch.var(local_batch, dim=1))[:,None]
    return local_batch


def STD_cleaning(local_batch,threshold,local_labels):
    idxs = torch.nonzero(torch.std(local_batch, dim=1)<threshold)
    return local_batch[idxs].squeeze(dim=1), local_labels[idxs].squeeze(dim=1)


