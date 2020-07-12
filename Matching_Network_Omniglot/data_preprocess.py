import numpy as np
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import tqdm
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

class load_dataset(object):

    def __init__(self, filename='../data/data.npy'):
        self.filename = filename

    def load_data(self):
        """
        :param filename: the name of data
        :return: a dictionary {"train": x_train, "val": x_val, "test": x_test}
        """

        x = np.load(self.filename)  # shape=(1623, 20, 28, 28) type=np.array.
        self.n_classes = x.shape[0]  # total number of classes
        x = np.reshape(x, newshape=(
        x.shape[0], x.shape[1], 28, 28, 1))  # expand dimension from (x.shape[0],x.shape[1],28,28)

        np.random.shuffle(x)  # shuffle dataset
        x_train, x_val, x_test = x[:1200], x[1200:1411], x[1411:]  # divide dataset in to train, val,ctest

        x_train = self.__processes_noraml(x_train)
        x_val = self.__processes_noraml(x_val)
        x_test = self.__processes_noraml(x_test)

        # Defining dictionary of dataset
        datatset = {"train": x_train, "val": x_val, "test": x_test}
        return datatset

    def __processes_noraml(self, data):
        """
        # Normalize Dataset
        :param data:
        :return: (data - mu) / sigma
        """
        mu = np.mean(data)
        sigma = np.std(data)
        return (data - mu) / sigma


class data_pre(object):
    def __init__(self, batch_size=16, classes_per_set=20, samples_per_class=1):
        self.batch_size = batch_size  # setting batch_size
        self.classes_per_set = classes_per_set  # Number of classes per set
        self.samples_per_class = samples_per_class  # as we are choosing it to be one shot learning, so we have 1 sample

    def sample_batch(self, data):
        """
        Generates support_set and  query_set
        :param : data - one of(train,test,val) our current dataset shape [total_classes,20,28,28,1]
        :return: [support_set_x,support_set_y,target_x,target_y] for Matching Networks
        """
        support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data.shape[2],
                                  data.shape[3], data.shape[4]), np.float32)
        support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), np.int32)

        query_set_x = np.zeros((self.batch_size, data.shape[2], data.shape[3], data.shape[4]), np.float32)
        query_set_y = np.zeros((self.batch_size, 1), np.int32)
        for i in range(self.batch_size):
            # For support_set: 1.choosing random classes(W way) 2. choosing random sample (K+1 shot)
            choose_classes = np.random.choice(data.shape[0], size=self.classes_per_set, replace=False)
            choose_samples = np.random.choice(data.shape[1], size=self.samples_per_class + 1, replace=False)

            x_temp = data[choose_classes]  # choosing classes
            x_temp = x_temp[:, choose_samples]  # choosing sample batch from classes chosen outputs 20X2X28X28X1

            y_temp = np.arange(self.classes_per_set)  # will return [0,1,2,3,...,19]
            support_set_x[i] = x_temp[:, :-1]
            support_set_y[i] = np.expand_dims(y_temp[:], axis=1)  # expand dimension

            # For query_set: every support_set have one example for query set
            choose_label = np.random.choice(self.classes_per_set, size=1)  # label set
            query_set_x[i] = x_temp[choose_label, -1]
            query_set_y[i] = y_temp[choose_label]
        # returns support of [batch_size, W way, k sample, 28, 28,1]; [batch_size, W way, 1 sample]
        # returns query   of [batch_size, 28, 28,1 ]; [batch_size, 1]
        return support_set_x, support_set_y, query_set_x, query_set_y

    def get_batch(self, name_data):
        """
        gen batch while training
        :param name_data: The data of dataset(one of "train","val","test")
        :return: a batch images
        """
        # W_way * k_sample
        support_set_x, support_set_y, query_set_x, query_set_y = self.sample_batch(name_data)
        support_set_x = support_set_x.reshape((support_set_x.shape[0], support_set_x.shape[1] * support_set_x.shape[2],
                                               support_set_x.shape[3], support_set_x.shape[4], support_set_x.shape[5]))
        support_set_y = support_set_y.reshape(support_set_y.shape[0], support_set_y.shape[1] * support_set_y.shape[2])
        return support_set_x, support_set_y, query_set_x, query_set_y

if __name__ == "__main__":

    all_dataset = load_dataset(filename='../data/data.npy').load_data()
    preprocess_data = data_pre(batch_size=16, classes_per_set=20, samples_per_class=1)
    x_support_set, y_support_set, query_set_x, query_set_y = preprocess_data.get_batch(name_data=all_dataset['train'])

