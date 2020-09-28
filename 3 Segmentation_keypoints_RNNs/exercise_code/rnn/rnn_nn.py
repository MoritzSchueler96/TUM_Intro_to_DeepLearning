import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, activation="tanh"):
        super().__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        #######################################################################
        # TODO: Build a simple one layer RNN with an activation with the      #
        # attributes defined above and a forward function below. Use the      #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h as 0 if these values are not given.                     #
        #######################################################################
        self.fc_x = nn.Linear(input_size, hidden_size)
        self.fc_h = nn.Linear(hidden_size, hidden_size)
        if activation == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU()
        self.h = torch.zeros([1, 1, hidden_size])
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################
        h_seq = torch.tensor([])
        h = self.h

        for seq in range(x.size(0)):
            h = self.act(self.fc_h(h) + self.fc_x(x[seq]))
            h_seq = torch.cat((h_seq, h), 0)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, h


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=20):
        super().__init__()
        #######################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes #
        # defined above and a forward function below. Use the                 #
        # nn.Linear() function as your linear layers.                         #
        # Initialize h and c as 0 if these values are not given.               #
        #######################################################################
        self.h = torch.zeros([1, 1, hidden_size])
        self.c = torch.zeros([1, 1, hidden_size])

        self.forget_x = nn.Linear(input_size, hidden_size)
        self.input1_x = nn.Linear(input_size, hidden_size)
        self.input2_x = nn.Linear(input_size, hidden_size)
        self.output_x = nn.Linear(input_size, hidden_size)

        self.forget_h = nn.Linear(hidden_size, hidden_size)
        self.input1_h = nn.Linear(hidden_size, hidden_size)
        self.input2_h = nn.Linear(hidden_size, hidden_size)
        self.output_h = nn.Linear(hidden_size, hidden_size)

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None, c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = None
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################

        """
        # better understanding version
        h_seq = torch.tensor([])
        
        for seq in range(x.size(0)):
            forget_gate = self.sig(self.forget_h(self.h) + self.forget_x(x[seq]))
            input_gate1 = self.sig(self.input1_h(self.h) + self.input1_x(x[seq]))
            input_gate2 = self.tanh(self.input2_h(self.h) + self.input2_x(x[seq]))
            input_gate = input_gate1 * input_gate2
            cell_gate = self.c * forget_gate + input_gate
            out_gate = self.tanh(cell_gate) * self.sig(self.output_h(self.h) + self.output_x(x[seq]))

            h = out_gate
            c = cell_gate
            self.c = c
            self.h = h
            h_seq = torch.cat((h_seq, h), 0)
        """

        # more efficient version
        h_seq = torch.tensor([])
        c = self.c
        h = self.h

        for seq in range(x.size(0)):
            forget_gate = self.sig(self.forget_h(h) + self.forget_x(x[seq]))
            input_gate1 = self.sig(self.input1_h(h) + self.input1_x(x[seq]))
            input_gate2 = self.tanh(self.input2_h(h) + self.input2_x(x[seq]))
            c = c * forget_gate + input_gate1 * input_gate2
            h = self.tanh(c) * self.sig(self.output_h(h) + self.output_x(x[seq]))

            h_seq = torch.cat((h_seq, h), 0)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, (h, c)


class RNN_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128,
                 activation="relu"):
        super(RNN_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a RNN classifier                                       #
        #######################################################################
        # rnn model
        self.rnn = nn.RNN(input_size, hidden_size)
        # fc net with output_size = num_classes
        self.dense = nn.Linear(hidden_size, classes)
        # set activation
        if activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Tanh()
        # set classifier
        self.soft = nn.Softmax(dim=1)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):

        # prediction = last hidden layer
        _, x1 = self.rnn(x)
        x2 = x1[0]
        # apply activation
        x3 = self.act(x2)
        # apply fc layer to get number of classes
        x4 = self.dense(x3)
        # apply classifier
        x5 = self.soft(x4)
        x = x5

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


class LSTM_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################
        # lstm model
        self.lstm = nn.LSTM(input_size, hidden_size * 2, dropout=0.4, num_layers=2)
        # fc net with output_size = num_classes
        self.dense = nn.Linear(hidden_size * 2, classes)
        # set classifier
        self.soft = nn.Softmax(dim=1)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def forward(self, x):
        # prediction = last hidden layer
        _, (x1, _) = self.lstm(x)
        x2 = x1[0]
        # apply fc layer to get number of classes
        x3 = self.dense(x2)
        # apply classifier
        x4 = self.soft(x3)
        x = x4

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
