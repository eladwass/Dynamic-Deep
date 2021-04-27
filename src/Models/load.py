from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
# import tensorflow.keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import random
import scipy.io as sio
import tqdm

STEP = 256

def data_generator(batch_size, preproc, x, y):
    num_examples = len(x)
    examples = zip(x, y)
    examples = sorted(examples, key = lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size]
                for i in range(0, end, batch_size)]
    random.shuffle(batches)
    while True:
        for batch in batches:
            x, y = zip(*batch)
            yield preproc.process(x, y)

class Preproc:

    def __init__(self, ecg, labels,app=True,skip=False,CINC=False):
        self.mean, self.std = compute_mean_std(ecg)
        def generage_classes(labels):
            classes = set()
            for label in labels:
                if isinstance(label,list):
                    for l in label:
                        classes.add(l)
                else:
                    classes.add(label)
            return sorted(classes)
        self.classes = generage_classes(labels) #sorted(set(l for label in labels for l in label))
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}
        self.app = app
        self.skip = skip
        self.CINC=CINC


    def process(self, x, y):
        if self.skip is True and self.app is True:
            return self.process_x(x), [self.process_x(x), self.process_x(x), self.process_y(y), self.process_y(y)]  # TODO change to 2 outputs
        elif self.app:
            return self.process_x(x),[self.process_x(x), self.process_y(y)] # TODO change to 2 outputs
        else:
            return self.process_x(x), self.process_x(x)  # TODO change to 2 outputs

    def process_x(self, x):
        x = pad(x)
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        return x

    def process_y(self, y):
        # TODO, awni, fix hack pad with noise for cinc
        if self.CINC:
            y = pad([[self.class_to_int[c] for c in s] for s in y], val=3, dtype=np.int32)

        y = to_categorical(
                y, num_classes=len(self.classes))
        return y

def pad(x, val=0, dtype=np.float32):
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded

def compute_mean_std(x):
    x = np.hstack(x)
    return (np.mean(x).astype(np.float32),
           np.std(x).astype(np.float32))

def load_dataset(data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []; ecgs = []
    for d in tqdm.tqdm(data):
        loaded_ecg = load_ecg(d['ecg'])
        if (loaded_ecg.shape[0]==8960):
            labels.append(d['labels'])
            ecgs.append(loaded_ecg)
    return ecgs, labels

def load_ecg(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()
    else: # Assumes binary 16 bit integers
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)

    trunc_samp = STEP * int(len(ecg) / STEP)
    return ecg[:trunc_samp]

if __name__ == "__main__":
    data_json = "examples/cinc17/train.json"
    train = load_dataset(data_json)
    preproc = Preproc(*train)
    gen = data_generator(32, preproc, *train)
    for x, y in gen:
        print(x.shape, y.shape)
        break
