import numpy as np
import argparse
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
from Matching_Network_Omniglot.data_preprocess import load_dataset, data_pre


def convLayer(in_channels, out_channels, dropout_prob=0.0):
    """
    :param dataset_name: The name of dataset(one of "train","val","test")
    :return: a batch images
    """
    cnn_seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(dropout_prob)
    )
    return cnn_seq

class Embeddings_extractor(nn.Module):
    def __init__(self, layer_size=64, num_channels=1, dropout_prob=0.5, image_size=28):
        super(Embeddings_extractor, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
        self.layer1 = convLayer(num_channels, layer_size, dropout_prob)
        self.layer2 = convLayer(layer_size, layer_size, dropout_prob)
        self.layer3 = convLayer(layer_size, layer_size, dropout_prob)
        self.layer4 = convLayer(layer_size, layer_size, dropout_prob)

        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size

    def forward(self, image_input):
        """
        :param: Image
        :return: embeddings
        """
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size()[0], -1)
        return x

class AttentionalClassify(nn.Module):
    """
    Create an Attention model after classifier
    a(x,x^)= softmax of cosine similarities
    """
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):
        """
        Products pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarites of size[batch_size,sequence_length]
        :param support_set_y:[batch_size,sequence_length,classes_num]
        :return: Softmax pdf shape[batch_size,classes_num]
        """
        softmax = nn.Softmax(dim=1)
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        return preds

class DistanceNetwork(nn.Module):
    """
    This model calculates the cosine distance between each of the support set embeddings and
    the target image embeddings.
    find cosine similarities between support set and input_test_image
    """

    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):
        """
        forward pass
        :param support_set:the embeddings of the support set images.shape[sequence_length,batch_size,64]
        :param input_image: the embedding of the target image,shape[batch_size,64]
        :return:shape[batch_size,sequence_length]
        """
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
            dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_manitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities.t()

class BidirectionalLSTM(nn.Module):
    """
    Create a Bi-directional LSTM, which is taking input and output from Test-image,
    and put them in same embeddings space.
    If we wish to use full-context embeddings, Matching Networks introduced Bi-directional LSTM for it.
    """
    def __init__(self, layer_size, batch_size, vector_dim):
        super(BidirectionalLSTM, self).__init__()
        """
        Initial a muti-layer Bidirectional LSTM
        :param layer_size: a list of each layer'size
        :param batch_size: 
        :param vector_dim: 
        """
        self.batch_size = batch_size
        self.hidden_size = layer_size[0]
        self.vector_dim = vector_dim
        self.num_layer = len(layer_size)
        self.lstm = nn.LSTM(input_size=self.vector_dim, num_layers=self.num_layer, hidden_size=self.hidden_size,
                            bidirectional=True)
        self.hidden = (
        Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size), requires_grad=False),
        Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size), requires_grad=False))

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables,
        to detach them from their history."""
        if type(h) == torch.Tensor:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, inputs):
        self.hidden = self.repackage_hidden(self.hidden)
        output, self.hidden = self.lstm(inputs, self.hidden)
        return output

class MatchingNetwork(nn.Module):
    def __init__(self, args):
        """
        Matching Network
        :param keep_prob: dropout rate
        :param batch_size:
        :param num_channels:
        :param learning_rate:
        :param fce: Flag indicating whether to use full context embeddings(i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set:
        :param num_samples_per_class:
        :param image_size:
        """
        super(MatchingNetwork, self).__init__()
        self.batch_size = args.batch_size
        self.keep_prob = args.keep_prob
        self.num_channels = args.num_channels
        self.learning_rate = args.lr
        self.num_classes_per_set = args.classes_per_set
        self.samples_per_class = args.samples_per_class
        self.image_size = args.image_size
        # Let's set all peices of Matching Networks Architecture
        self.g = Embeddings_extractor(layer_size=64, num_channels=args.num_channels, dropout_prob=args.keep_prob,
                                      image_size=args.image_size)
        self.f = args.fce  # if we are considering full-context embeddings
        self.c = DistanceNetwork()  # cosine distance among embeddings
        self.a = AttentionalClassify()  # softmax of cosine distance of embeddings
        if self.f:
            self.lstm = BidirectionalLSTM(layer_size=[32], batch_size=self.batch_size, vector_dim=self.g.outSize)

    def forward(self, support_set_images, support_set_y_one_hot, target_image, target_y):
        """
        Main process of the network
        :param support_set_images: shape[batch_size,sequence_length,num_channels,image_size,image_size]
        :param support_set_y_one_hot: shape[batch_size,sequence_length,num_classes_per_set]
        :param target_image: shape[batch_size,num_channels,image_size,image_size]
        :param target_y:
        :return:
        """
        # produce embeddings for support set images
        encoded_images = []
        for i in np.arange(support_set_images.size(1)):
            gen_encode = self.g(support_set_images[:, i, :, :])
            encoded_images.append(gen_encode)

        # produce embeddings for target images
        gen_encode = self.g(target_image)
        encoded_images.append(gen_encode)
        output = torch.stack(encoded_images, dim=0)

        # if we are considering full-context embeddings
        if self.f:
            output = self.lstm(output)

        # get similarities between support set embeddings and target
        similarites = self.c(support_set=output[:-1], input_image=output[-1])

        # produce predictions for target probabilities
        preds = self.a(similarites, support_set_y=support_set_y_one_hot)

        # calculate the accuracy
        values, indices = preds.max(1)
        accuracy = torch.mean((indices.squeeze() == target_y).float())
        crossentropy_loss = F.cross_entropy(preds, target_y.long())

        return accuracy, crossentropy_loss

def run_epoch(total_train_batches, args, name='train'):
    """
    using Omniglot Dataset,it will create a Omnligloat builder which calls Matching Network,
    and run its epochs for training, testing, and validation purpose.
    :param total_train_batches: Number of batches to train on
    :return:
    """
    total_c_loss = 0.0
    total_accuracy = 0.0
    for i in range(int(total_train_batches)):
            # x_support_set, y_support_set, x_target, y_target = get_batch(name)
            x_support_set, y_support_set, query_set_x, query_set_y = preprocess_data.get_batch(
                name_data=all_dataset[name])
            x_support_set = Variable(torch.from_numpy(x_support_set)).float()
            y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
            x_target = Variable(torch.from_numpy(query_set_x)).float()
            y_target = Variable(torch.from_numpy(query_set_y), requires_grad=False).squeeze().long()

            # convert to one hot encoding
            y_support_set = y_support_set.unsqueeze(2)
            sequence_length = y_support_set.size()[1]
            batch_size = y_support_set.size()[0]
            y_support_set_one_hot = Variable(
                torch.zeros(batch_size, sequence_length,
                            args.classes_per_set).scatter_(2,y_support_set.data,1), requires_grad=False)

            # reshape channels and change order
            size = x_support_set.size()
            x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
            x_target = x_target.permute(0, 3, 1, 2)
            acc, c_loss = matchNet(x_support_set, y_support_set_one_hot, x_target, y_target)

            # optimize process
            optimizer.zero_grad()
            c_loss.backward()
            optimizer.step()

            iter_out = "tr_loss: {}, tr_accuracy: {}".format(c_loss, acc)
            total_c_loss += c_loss
            total_accuracy += acc

    total_c_loss = total_c_loss / total_train_batches
    total_accuracy = total_accuracy / total_train_batches
    return total_c_loss, total_accuracy

parser = argparse.ArgumentParser(description="The parameter of Match_NN")
parser.add_argument("--batch_size", type=int, help="Batch of support set", default=11)
parser.add_argument("--num_channels", type=int, help="channel of image", default=1)
parser.add_argument("--image_size", type=int, help="size of image", default=28)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
parser.add_argument("--classes_per_set", type=int, help="w way classes of per support set", default=20)
parser.add_argument("--samples_per_class", type=int, help="samples k shot per support set", default=1)
parser.add_argument("--keep_prob", type=float, help="parameter of dropout - keep_prob", default=0.0)
parser.add_argument("--fce", type=bool, help="if considere full-context embeddings", default=True)

args = parser.parse_args()

# batch_size=10
# num_channels=1
# lr=1e-3
# image_size=28
# classes_per_set=20
# samples_per_class=1
# keep_prob=0.0
# fce=True
optim="adam"
wd=0

all_dataset = load_dataset(filename='../data/data.npy').load_data()
preprocess_data = data_pre(batch_size=args.batch_size, classes_per_set=args.classes_per_set, samples_per_class=args.samples_per_class)

matchNet = MatchingNetwork(args)
total_iter = 0
total_train_iter = 0
optimizer = torch.optim.Adam(matchNet.parameters(), lr=args.lr, weight_decay=wd)
scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True)
# Training setup
total_epochs = 10
total_train_batches = 10
total_val_batches = 5
total_test_batches = 5
train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
test_loss, test_accuracy = [], []

for e in range(total_epochs):
    ############################### Training Step ##########################################
    total_c_loss, total_accuracy = run_epoch(total_train_batches, args, 'train')
    train_loss.append(total_c_loss)
    train_accuracy.append(total_accuracy)

    ################################# Validation Step #######################################
    total_val_c_loss, total_val_accuracy = run_epoch(total_val_batches, args, 'val')
    val_loss.append(total_val_c_loss)
    val_accuracy.append(total_val_accuracy)
    print("Epoch {}: train_loss:{:.2f} train_accuracy:{:.2f} valid_loss:{:.2f} valid_accuracy:{:.2f}".
          format(e, total_c_loss, total_accuracy, total_val_c_loss, total_val_accuracy))


total_test_c_loss, total_test_accuracy = run_epoch(total_test_batches,args, 'test')
print("test_accuracy:{}%".format(total_test_accuracy*100))
def plot_loss(train,val,name1="train_loss",name2="val_loss",title=""):
    plt.title(title)
    plt.plot(train, label=name1)
    plt.plot(val, label=name2)
    plt.show()

plot_loss(train_loss,val_loss,"train_loss","val_loss","Loss Graph")
plot_loss(train_accuracy,val_accuracy,"train_accuracy","val_accuracy","Accuracy Graph")