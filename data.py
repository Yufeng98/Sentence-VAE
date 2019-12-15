import os
import io
import json
import numpy as np
from random import shuffle
from sklearn.preprocessing import scale
from torch.utils.data import Dataset


class Data(Dataset):

    def __init__(self, split, batch_size, data_dir, subject, seq_len, embedding_size, cut_start, lines):

        super().__init__()
        self.split = split
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.subject = subject
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.cut_start = cut_start
        self.lines = lines
        self.dir = os.getcwd()
        if not os.path.exists(self.dir + '/{}_data.json'.format(self.data_dir)):
            print("{} preprocessed file not found. Creating new.".format(self.data_dir))
            self._create_data()
        else:
            self._load_data()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return np.array(self.dataset[idx])

    def _load_data(self):
        with open(self.dir + '/{}_data.json'.format(self.data_dir), 'r') as file:
            self.dataset = np.array(json.load(file)[self.split])

    def _create_data(self):

        file_dir = '/{}'.format(self.data_dir)
        file = list(os.walk(self.dir + file_dir))[0][-1]
        data_norm = []
        data_dict = {}
        dataset = {}

        for filename in file:

            # read raw data
            f = open(self.dir + "/{}/{}".format(self.data_dir, filename))
            lines_str = f.readlines()[self.cut_start: self.cut_start + self.lines]
            f.close()
            lines = []
            for i in range(self.lines):
                lines.append(list(map(float, lines_str[i].split())))            # UM lines [288, 200]

            # truncation from self.lines to [self.seq_len, self.embedding_size]
            subject = []
            for j in range(self.seq_len):
                lst = []
                for i in lines[self.embedding_size * j: self.embedding_size * (j + 1)]:
                    lst.append(np.array(list(i)))
                subject.append(np.array(lst))
            subject = np.array(subject)                                         # UM subject [9, 32, 200]
            data_norm.append(subject)                                           # UM data_norm [95, 9, 32, 200]

        # normalization for each line
        data_norm = np.array(data_norm)
        data_norm_swap = []
        for i in range(200):
            data_norm_swap.append(data_norm[:, :, :, i])                        # UM data_norm_swap [200, 95, 9, 32]
        # tmp = np.swapaxes(data_norm, 2, 3)
        # tmp = np.swapaxes(tmp, 1, 2)
        # tmp = np.swapaxes(tmp, 0, 1)

        # normalize for each subject
        # data_norm_swap = np.swapaxes(data_norm_swap, 0, 1)                      # UM data_norm_swap [95, 200, 9, 32]

        # normalize for each node
        data_norm_swap = np.array(data_norm_swap).reshape((200, -1))
        # data_norm_swap = np.array(data_norm_swap).reshape(-1)
        data_mean = np.mean(data_norm_swap, axis=1)
        data_std = np.std(data_norm_swap, axis=1)
        data_normalization = np.array([(data_norm_swap[i] - data_mean[i]) / data_std[1] for i in range(200)])
        # data_normalization = np.array([data_norm_swap - data_mean]) / data_std
        # data_normalization = data_normalization.reshape((95, 200, 9, 32))
        data_normalization = data_normalization.reshape((200, self.subject, self.seq_len, self.embedding_size))

        for i in range(len(file)):
            data_dict[file[i]] = np.array(data_normalization[:, i, :, :])       # UM data_dict [95, 200, 9, 32]
            # data_dict[file[i]] = np.array(data_normalization[i, :, :, :])       # UM data_dict [95, 200, 9, 32]

        # shuffle to split training and validation sets
        data_list = list(data_dict.items())
        shuffle(data_list)
        data_shuffle = []
        for i in range(self.subject):
            data_shuffle.append([i]+list(data_list[i]))                        # UM [No., filename, [200,9,32]] * 95
        data = [i[-1] for i in data_shuffle]

        # reshape data and build dataset
        data = np.array(data).reshape(self.subject*(200//self.batch_size),
                                      self.batch_size, self.seq_len, self.embedding_size)   # UM [950, 20, 9, 32]
        train_data = data[:round(self.subject*0.8)*(200//self.batch_size), :, :, :]         # UM train [760, 20, 9, 32]
        valid_data = data[round(self.subject*0.8)*(200//self.batch_size):, :, :, :]         # UM valid [190, 20, 9, 32]
        dataset['train'] = train_data.reshape(-1, self.seq_len, self.embedding_size).tolist()
        dataset['valid'] = valid_data.reshape(-1, self.seq_len, self.embedding_size).tolist()
        dataset['Order'] = [i[:2] for i in data_shuffle]

        # write data to '/data.json'
        with io.open(self.dir + '/{}_data.json'.format(self.data_dir), 'wb') as data_file:
            data = json.dumps(dataset, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data()
