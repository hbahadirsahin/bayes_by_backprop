import numpy as np
import os
import pickle as pickle

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = ROOT_DIR + "/cifar_data/"



def load_dataset():
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(DATA_PATH, 'data_batch_%d' % (b,))
        X, Y = get_features_and_labels(f)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    Y_train = np.concatenate(ys)
    del X, Y
    X_test, Y_test = get_features_and_labels(os.path.join(DATA_PATH, 'test_batch'))
    return normalize(X_train), one_hot_encoded(Y_train, 10), normalize(X_test), one_hot_encoded(Y_test, 10)


def get_features_and_labels(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def one_hot_encoded(class_numbers, num_classes=None):
    if num_classes is None:
        num_classes = np.max(class_numbers) - 1

    return np.eye(num_classes, dtype=float)[class_numbers]


def normalize(x):
    maximum = np.max(x)
    minimum = np.min(x)
    return (x - minimum) / (maximum - minimum)
