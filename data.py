import os
import io
import json
import numpy as np
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
            name = '/{}_data.json'.format(self.data_dir)
            print("{} preprocessed file not found at {}. Creating new.".format(split.upper(), self.dir + name))
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

        file_dir = './{}'.format(self.data_dir)
        file = list(os.walk(file_dir))[0][-1]
        data = []
        dataset = {}

        for filename in file:

            # read raw data
            f = open("./{}/{}".format(self.data_dir, filename))
            lines_str = f.readlines()[self.cut_start: self.cut_start + self.lines]
            f.close()

            # normalization for each line
            lines = []
            for i in range(self.lines):
                lines.append(scale(list(map(float, lines_str[i].split()))))

            # truncation from self.lines to [self.seq_len, self.embedding_size]
            subject = []
            for j in range(self.seq_len):
                lst = []
                for i in lines[self.embedding_size * j: self.embedding_size * (j + 1)]:
                    lst.append(np.array(list(i)))
                subject.append(np.array(lst))
            subject = np.array(subject)
            node = []
            for i in range(200):
                node.append(subject[:, :, i])
            data.append(np.array(node))                                                     # UM [95, 200, 9, 32]

        # reshape data and build dataset
        data = np.array(data).reshape(self.subject*(200//self.batch_size),
                                      self.batch_size, self.seq_len, self.embedding_size)   # UM [950, 20, 9, 32]
        train_data = data[:round(self.subject*0.8)*(200//self.batch_size), :, :, :]         # UM train [760, 20, 9, 32]
        valid_data = data[round(self.subject*0.8)*(200//self.batch_size):, :, :, :]         # UM valid [190, 20, 9, 32]
        dataset['train'] = train_data.reshape(-1, self.seq_len, self.embedding_size).tolist()
        dataset['valid'] = valid_data.reshape(-1, self.seq_len, self.embedding_size).tolist()

        # write data to '/data.json'
        with io.open(self.dir + '/{}_data.json'.format(self.data_dir), 'wb') as data_file:
            data = json.dumps(dataset, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data()
