from io import StringIO
import numpy as np
from six.moves import cPickle
import random

def extract_keystrokes(filename):
    """[stroke_id, key_a, key_b, diff]"""
    print('Extracting', filename)
    data = np.loadtxt(open(filename, 'rb'), delimiter=',', usecols=(1, 2, 3))
    return data

def dense_to_one_hot(labels_dense, num_classes=256):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def extract_labels(filename):
    """stroke_id [user]"""
    print('Extracting', filename)
    dt = np.dtype(np.uint32)
    labels = np.loadtxt(open(filename, 'rb'), delimiter=',', usecols=(0,), dtype=dt)
    return dense_to_one_hot(labels)

class DataSet(object):
    def __init__(self, keystrokes, labels):
        self.num_examples = keystrokes.shape[0]
        self.keystrokes = keystrokes
        self.labels = labels
        self.epochs_completed = 0
        self.index_in_epoch = -1

    def next_batch(self, batch_size):
        start = random.randint(0, self.num_examples-batch_size-1)
        end = start + batch_size
        self.index_in_epoch += 1
        return self.keystrokes[start:end], self.labels[start:end]

def read_data_sets(fin):
    class DataSets(object):
        pass
    data_sets = DataSets()

    train_keystrokes = extract_keystrokes(fin)
    train_labels = extract_labels(fin)

    #TODO make actual training set
    test_keystrokes = train_keystrokes
    test_labels = train_labels

    validation_keystrokes = train_keystrokes[500:]
    validation_labels = train_labels[500:]

    data_sets.train = DataSet(train_keystrokes, train_labels)
    data_sets.validation = DataSet(validation_keystrokes, validation_labels)
    data_sets.test = DataSet(test_keystrokes, train_labels)

    return data_sets

#print(extract_labels('data/keystrokes.csv').shape)
#print(extract_keystrokes('data/keystrokes.csv').shape)
