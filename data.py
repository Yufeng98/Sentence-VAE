import os
import io
import json
import numpy as np
from sklearn.preprocessing import scale
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
from utils import OrderedCounter


class Data(Dataset):

    def __init__(self, split):

        super().__init__()
        self.split = split
        self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _load_data(self, vocab=True):

        file_dir = 'C://Users//Yufeng.Gu//Desktop//Yale//ABIDE//Outputs//ccs//filt_noglobal//rois_cc200//UM'
        file = list(os.walk(file_dir))[0][-1][:95]
        data = []
        for filename in file:
            # read raw data
            f = open("C://Users//Yufeng.Gu//Desktop//Yale//ABIDE//Outputs//ccs//filt_noglobal//rois_cc200//UM//"
                     "{}".format(filename))
            lines_str = f.readlines()[4:-4]
            f.close()
            # normalization
            lines = []
            for i in range(288):
                lines[i] = scale(list(map(float, lines_str[i].split())))
            # truncation
            subject = []
            for j in range(9):
                lst = []
                for i in lines[32 * j: 32 * (j + 1)]:
                    lst.append(np.array(list(i)))
                subject.append(np.array(lst))
            subject = np.array(subject)
            node = []
            for i in range(200):
                node.append(subject[:, :, i])
            data.append(np.array(node))
        data = np.array(data).reshape(950, 20, 9, 32)  # [95, 200, 9, 32]
        train_data = data[:800, 20, 9, 32]
        valid_data = data[800:, 20, 9, 32]
        if self.split == 'train':
            self.dataset = train_data.reshape(-1, 9, 32)
        elif self.split == 'valid':
            self.dataset = valid_data.reshape(-1, 9, 32)
