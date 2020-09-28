import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # TODO: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(1, 32, 4, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, 2, stride=1)
        self.conv4 = nn.Conv2d(128, 256, 1, stride=1)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        self.dense1 = nn.Linear(6400, 1000)
        self.dense2 = nn.Linear(1000, 1000)
        self.dense3 = nn.Linear(1000, 30)

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################

        x1 = self.pool1(F.elu(self.conv1(x)))
        x2 = self.dropout1(x1)
        x3 = self.pool2(F.elu(self.conv2(x2)))
        x4 = self.dropout2(x3)
        x5 = self.pool3(F.elu(self.conv3(x4)))
        x6 = self.dropout3(x5)
        x7 = self.pool4(F.elu(self.conv4(x6)))
        x8 = self.dropout4(x7)
        x9 = x8.view(x8.size(0), -1)  # flatten
        #x9 = x8.view(-1, 6400)
        x10 = self.dropout5(F.elu(self.dense1(x9)))
        x11 = self.dropout6(F.relu(self.dense2(x10)))  # linear activation
        x12 = self.dense3(x11)
        x = x12

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
