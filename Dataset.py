import torch
from torch.utils.data import Dataset
import numpy as np
import os
from utils import extract_cds_ncds

from hmm import extract_features

class GenDataset:
    def __init__ (self, cancer_dir, normal_dir):
        self.X = []
        self.Y = []

        self.load_data(cancer_dir, label=1)
        self.load_data(normal_dir, label=0)

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

    def load_data (self, folder, label):
        for file in os.listdir(folder):
            if file.endswith(".gb") or file.endswith(".gbk"):
                path = os.path.join(folder, file)

                cds, ncds = extract_cds_ncds(path)

                if len(cds) == 0 and len(ncds) == 0:
                    continue

                features = extract_features(cds, ncds)

                self.X.append(features)
                self.Y.append(label)

    def get_data(self):
        return self.X, self.Y
    
    def __len__(self):
        return len(self.X)

    def cross_validation_split (self, k=10):

        fold_size = self.len() // k
        indices = np.arrange(self.len())
        folds = []
        for i in range(k):
            test = indices[i * fold_size: (i + 1) * fold_size]
            train = np.concatenate([indices[:i * fold_size], indices[(i+1) * fold_size:]])
            folds.append((train, test))

        return folds
    
    def get_fold (self, train_index, test_index):

        train_X = [], train_Y = []
        test_X = [], test_Y = []
        for i in range(self.len()):
            if i in train_index:
                train_X.append(self.X[i])
                train_Y.append(self.Y[i])
            elif i in test_index:
                test_X.append(self.X[i])
                test_Y.append(self.Y[i])

        return train_X, train_Y, test_X, test_Y
       
    