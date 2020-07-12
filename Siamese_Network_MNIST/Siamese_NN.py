import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
from Siamese_Network_MNIST.data_preprocess import get_data,plot_figure

class SiameseNetwork(nn.Module):
    """
    Creating Siamese Network Architecture
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.MaxPool2d(2, stride=2))
        self.fc1 = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
            nn.Linear(10, 2))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    # Create Loss Function
    def contrastive_loss_function(self,x0, x1, y, margin=1.0):
        # euclidean distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

class train_and_test(object):
    def __init__(self,model):
        self.model = model
        self.plot = plot_figure()
        self.train_loss = []
        self.test_pre = []
        self.test_label = []

    def train(self, train_loader,  plot_train_loss=False):
        # Train Model for certain number of epochs.
        epochs = 100
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        for epoch in range(epochs):
            for batch_idx, (x0, x1, labels) in enumerate(train_loader):
                labels = labels.float()
                x0, x1, labels = Variable(x0), Variable(x1), Variable(labels)
                output1, output2 = self.model(x0, x1)
                loss = self.model.contrastive_loss_function(output1, output2, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.train_loss.append(loss.item())
                if batch_idx % 10 == 0:
                    print('Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
        if plot_train_loss == True:
            self.plot.plot_loss(self.train_loss)
    def test(self, test_loader, plot_test_pre=False):
        self.model.eval() # this will auto close dropout,BN.
        with torch.no_grad():

            for batch_idx, (x,labels) in enumerate(test_loader):
                x, labels = Variable(x), Variable(labels)
                pre = self.model.forward_once(x)
                self.test_pre.extend(pre.data.cpu().numpy().tolist())
                self.test_label.extend(labels.data.cpu().numpy().tolist())
        self.test_pre = np.array(self.test_pre)
        self.test_label = np.array(self.test_label)
        if plot_test_pre == True:
            self.plot.plot_Siamese_pre(self.test_pre,self.test_label)

if __name__ == "__main__":

    train_loader, test_loader = get_data() # Obtain Data

    train_and_test = train_and_test(model=SiameseNetwork())

    train_and_test.train(train_loader, plot_train_loss=True)
    train_and_test.test(test_loader, plot_test_pre=True)
