import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import deepdish as dd
import json
import csv
import os
import io

np.random.seed(5)
methods = ["svm", "rf"]
# parameter = [['rbf', 1e-4], ['rbf', 1e-3], ['rbf', 0.01], ['rbf', 0.1], ['rbf', 1],
#              ['rbf', 10], ['rbf', 1e2], ['rbf', 1e3], ['rbf', 1e4]]
parameter = [['rbf', 1]]
fold = os.getcwd()
# site, num = ["UM_correlation_matrix", 12]
# site, num = ["NYU_correlation_matrix", 11]
# site, num = ["UCLA_correlation_matrix", 14]
# site, num = ["USM_correlation_matrix", 11]
for site, num, seq_len, subject in [["USM_correlation_matrix", 11, 8, 52], ["UM_correlation_matrix", 12, 9, 88],
                                    ["NYU_correlation_matrix", 11, 7, 167], ["UCLA_correlation_matrix", 14, 7, 63]]:
    file_dir = fold + '/' + site
    file = list(os.walk(file_dir))[0][-1][:]
    label = []
    data = []
    rows = {}  # label: rows = ['UM_1_0050272':1 ...]

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
            # data.append(np.sum(tmp, axis=0))
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

    index = [i for i in range(subject)]
    np.random.shuffle(index)

    split_sub = []
    split_len = subject // 5
    for i in range(4):
        split_sub.append(index[i * split_len: (i + 1) * split_len])
    split_sub.append(index[4 * split_len:])

    split = []
    for i in range(4):
        tmp = []
        for j in split_sub[i]:
            for k in range(seq_len):
                tmp.append(j * seq_len + k)
        split.append(tmp)
    tmp = []
    for j in split_sub[4]:
        for k in range(seq_len):
            tmp.append(j * seq_len + k)
    split.append(tmp)

    acc_split = []
    accuracy = {"rf": 0, "svm": 0}
    accuracy_split = {"rf": [], "svm": []}
    for k in range(5):
        sum_acc = 0
        split_train = [0, 1, 2, 3, 4]
        split_train.remove(k)
        train_index = []
        for j in split_train:
            train_index += split[j]
        test_index = split[k]
        train_data = data[train_index]
        mean = train_data.mean(0)
        dev = train_data.std(0)
        train_data = (train_data - mean) / dev
        train_label = label[train_index]
        test_data = data[test_index]
        test_label = label[test_index]
        test_data = (test_data - mean) / dev
        for j in methods:
            if j == "rf":
                clf = RandomForestClassifier(n_estimators=1000, max_depth=3)
            else:
                clf = svm.SVC(kernel='rbf', C=1, gamma='auto', probability=False)
            clf.fit(train_data, train_label)
            label_pred = clf.predict(test_data)
            # print(sum(label_pred))
            # label_pred_prob = clf.predict_proba(test_data)
            # label_pred_log_prob = clf.predict_log_proba(test_data)
            pred = test_label == label_pred
            predict = {}
            for i in range(len(test_index)):
                subject_id = test_index[i]//seq_len
                if subject_id not in predict.keys():
                    predict[subject_id] = [pred[i]]
                else:
                    predict[subject_id].append(pred[i])
            threshold = (seq_len - 1) // 2
            true = 0
            for key in predict.keys():
                if sum(predict[key]) >= (seq_len - threshold):
                    true += 1
            accuracy[j] = true / len(test_index) * seq_len
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
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    pred = y == y_pred
    print(sum(pred))
