from torchvision import transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

class MNIST_Siamese_Preprocess(object):
    def __init__(self,data, target):
        self.mnist_data = data
        self.mnist_target = target
        self.digit_indices = [np.where(self.mnist_target == i)[0] for i in range(10)] # -> [[indices],...,[indices]] (len=10)
        self.min_len_digit_indices = min([len(self.digit_indices[d]) for d in range(10)]) - 1

        self.create_pairs()

    def create_pairs(self):
        """
        Before creating an iterator, lets create pairs, and preprocess images in them
        """

        x0_data = []
        x1_data = []
        label = []

        for num in range(10):  # for MNIST dataset: as we have 10 digits
            for i in range(self.min_len_digit_indices):
                z1, z2 = self.digit_indices[num][i], self.digit_indices[num][i + 1]
                x0_data.append(self.mnist_data[z1] / 255.)  # Image Preprocessing Step
                x1_data.append(self.mnist_data[z2] / 255.)  # Image Preprocessing Step
                label.append(1)

                dn = (num + random.randrange(1, 10)) % 10   # get different num data
                z1, z2 = self.digit_indices[num][i], self.digit_indices[dn][i]
                x0_data.append(self.mnist_data[z1] / 255.)  # Image Preprocessing Step
                x1_data.append(self.mnist_data[z2] / 255.)  # Image Preprocessing Step
                label.append(0)


        self.x0 = torch.from_numpy(np.array(x0_data, dtype=np.float32).reshape([-1, 1, 28, 28]))
        self.x1 = torch.from_numpy(np.array(x1_data, dtype=np.float32).reshape([-1, 1, 28, 28]))
        self.label = torch.from_numpy(np.array(label, dtype=np.int32))
        self.size = np.array(label, dtype=np.int32).shape[0]
        return None

    def __getitem__(self, index):
        return (self.x0[index],
                self.x1[index],
                self.label[index])

    def __len__(self):
        return self.size

class plot_figure(object):
    def plot_loss(self, loss_list, name="train_loss"):
        """
        :param loss_list: input list
        :param name: figure name
        :return: figure
        """
        plt.plot(loss_list,label=name)
        plt.show()
    def plot_Siamese_pre(self, test_pre, test_label):
        """
        :param test_pre: The embedding_out Of Siamese Network(forward_once) shape=(,2).
        :param test_label: real label of 0-9
        :return:
        """
        color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        for i in range(10):
            f = test_pre[np.where(test_label == i)]
            plt.plot(f[:, 0], f[:, 1], '.', c=color[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.show()

def get_data():
    train = dsets.MNIST(root='../data/', train=True, download=True)
    test = dsets.MNIST(root='../data/', train=False, transform=transforms.Compose([transforms.ToTensor(), ]))
    indices = np.random.choice(len(train.targets.numpy()), 2000, replace=False)
    indices_test = np.random.choice(len(test.targets.numpy()), 100, replace=False)

    # We created an iterator above, here we will use it to create training and test set iterators.
    train_iter = MNIST_Siamese_Preprocess(train.data.numpy()[indices], train.targets.numpy()[indices])
    # test_iter = MNIST_Siamese_Preprocess(test.data.numpy()[indices_test], test.targets.numpy()[indices_test])

    # creating a train loader, and a test loader.
    train_loader = torch.utils.data.DataLoader(train_iter, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)
    return train_loader, test_loader

if __name__ == "__main__":

    train_loader, test_loader = get_data()  # Obtain Data


    for batch_idx, (x0, x1, labels) in enumerate(train_loader):

        print("batch_idx {}, x0_shape {} x1_shape {}".format(batch_idx, x0.size(), x1.size()))

    for batch_idx , (x, labels) in enumerate(test_loader):
        print("x_shape {} labels {}".format(x.size(), labels.size()))