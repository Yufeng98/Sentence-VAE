import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import deepdish as dd
import random
import json
import csv
import os
import io

np.random.seed(0)
methods = ["svm", "rf"]
parameter = [i for i in range(2, 5)]
# site = "UM_correlation_matrix"
# site = "NYU_correlation_matrix"
# site = "USM_correlation_matrix"
# site = "UCLA_correlation_matrix"

for site, num in [["UM_correlation_matrix", 12], ["NYU_correlation_matrix", 11],
                  ["UCLA_correlation_matrix", 14], ["USM_correlation_matrix", 11]]:
    fold = os.getcwd()
    file_dir = fold + '/' + site
    file = list(os.walk(file_dir))[0][-1][:]
    label = []
    data = []
    rows = {}       # label: rows = ['UM_1_0050272':1 ...]

    if not os.path.exists(fold + '/{}.h5'.format(site)):
        with open(fold + '/Phenotypic_V1_0b_preprocessed1.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for i in reader:
                if list(i[0].split(','))[5] in ['UM_1', 'NYU', 'USM', 'UCLA_1']:
                    name, lab = list(i[0].split(','))[6:8]
                    lab = int(lab)
                    rows[name] = lab
        for filename in file:
            tmp = dd.io.load(fold + "//{}//{}".format(site, filename))
            tri = np.triu(tmp, 1).reshape(-1)
            tri = tri[tri != 0]
            tri[tri < 0] = 0
            data.append(tri)
            label.append(int(rows[filename[:num]]) % 2)
        data = np.array(data)
        label = np.array(label)
        dataset = {'data': data, 'label': label}
        dd.io.save(fold + '/{}.h5'.format(site), dataset)
        data = np.array(dataset['data'])
        label = np.array(dataset['label'])
    else:
        dataset = dd.io.load(fold + '/{}.h5'.format(site))
        data = np.array(dataset['data'])
        label = np.array(dataset['label'])

    index = [i for i in range(data.shape[0])]
    np.random.shuffle(index)
    split = []
    split_len = data.shape[0] // 5
    for i in range(4):
        split.append(index[i*split_len: (i+1)*split_len])
    split.append(index[4*split_len:])

    acc_split = []
    accuracy = {}
    accuracy_split = {"rf": [], "svm": []}
    for k in range(5):
        split_train = [0, 1, 2, 3, 4]
        split_train.remove(k)
        train_index = []
        for j in split_train:
            train_index += split[j]
        test_index = split[k]
        train_data = data[train_index]
        train_label = label[train_index]
        test_data = data[test_index]
        test_label = label[test_index]
        for j in methods:
            if j == "rf":
                clf = RandomForestClassifier(n_estimators=1000, max_depth=3)
            else:
                clf = svm.SVC(kernel='rbf', C=1, gamma='auto')
            clf.fit(train_data, train_label)
            label_pred = clf.predict(test_data)
            print(sum(label_pred))
            pred = test_label == label_pred
            accuracy[j] = sum(pred) / len(label_pred)
        accuracy_split["rf"].append(accuracy["rf"])
        accuracy_split["svm"].append(accuracy["svm"])
        # print("{} {}:{}".format(site, k, average_acc))
    print("{} SVM acc: {}".format(site, sum(accuracy_split["svm"]) / 5))
    print("{} Random Foreast acc: {}".format(site, sum(accuracy_split["rf"]) / 5))

test = False
if test:
    y = [np.random.randint(2) for i in range(40)]
    X = np.random.rand(40, 3)
    # fit the model, don't regularize for illustration purposes
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    pred = y == y_pred
    print(sum(pred))
    print("1")