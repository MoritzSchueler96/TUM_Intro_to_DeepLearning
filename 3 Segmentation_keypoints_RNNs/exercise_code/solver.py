import numpy as np
import torch
import math


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data_train.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data_train in torch.utils.data_train.DataLoader
        - val_loader: val data_train in torch.utils.data_train.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        #######################################################################
        # TODO:                                                               #
        # Write your own personal training method for our solver. In each     #
        # epoch iter_per_epoch shuffled training batches are processed. The   #
        # loss for each batch is stored in self.train_loss_history. Every     #
        # log_nth iteration the loss is logged. After one epoch the training  #
        # accuracy of the last mini batch is logged and stored in             #
        # self.train_acc_history. We validate at the end of each epoch, log   #
        # the result and store the accuracy of the entire validation set in   #
        # self.val_acc_history.                                               #
        #                                                                     #
        # Your logging could like something like:                             #
        #   ...                                                               #
        #   [Iteration 700/4800] TRAIN loss: 1.452                            #
        #   [Iteration 800/4800] TRAIN loss: 1.409                            #
        #   [Iteration 900/4800] TRAIN loss: 1.374                            #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                           #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                           #
        #   ...                                                               #
        #######################################################################
        current_mini_batch_train = 0

        for epoch in range(0, num_epochs):

            # Training
            scores_train = []
            model.train()

            # iterate over training batches
            for batch_idx_train, (data_train, target_train) in enumerate(train_loader):
                # optimize model
                optim.zero_grad()
                output_train = model(data_train)
                #output_train = model.forward(data_train)
                loss_train = self.loss_func(output_train, target_train)
                loss_train.backward()
                optim.step()

                # get prediction and scores
                _, pred_train = torch.max(output_train, 1)
                targets_mask = target_train >= 0
                scores_train.append(np.mean((pred_train == target_train)[targets_mask].data.cpu().numpy()))
                self.train_loss_history.append(loss_train.item())

                # print loss each training step
                current_mini_batch_train += math.ceil(len(data_train)/10)*10
                if (batch_idx_train + 1) % log_nth == 0:
                    num_samples = num_epochs * iter_per_epoch * len(data_train)
                    print('[Iteration {}/{}]\tTRAIN Loss:\t{:.3f}'.format(current_mini_batch_train, num_samples, loss_train.item()))
            print('[Iteration {}/{}]\tTRAIN Loss:\t{:.3f}'.format(current_mini_batch_train, num_samples, loss_train.item()))

            # training stats
            train_acc = np.mean(scores_train)
            self.train_acc_history.append(train_acc)
            print('[Epoch {}/{}]\t\tTRAIN acc/loss:\t{:.3f}/{:.3f}'.format(
                (epoch+1), num_epochs, train_acc, loss_train.item()))

            # Validation
            scores_val = []
            model.eval()

            for batch_idx_val, (data_val, target_val) in enumerate(val_loader):
                # get validation loss
                output_val = model(data_val)
                #output_val = model.forward(data_val)
                loss_val = self.loss_func(output_val, target_val)

                # get prediction and scores
                _, pred_val = torch.max(output_val, 1)
                targets_mask = target_val >= 0
                scores_val.append(np.mean((pred_val == target_val)[targets_mask].data.cpu().numpy()))
                self.val_loss_history.append(loss_val.item())

            # validation stats
            val_acc = np.mean(scores_val)
            self.val_acc_history.append(val_acc)
            print('[Epoch {}/{}]\t\tVAL acc/loss:\t{:.3f}/{:.3f}'.format(
                (epoch+1), num_epochs, val_acc, loss_val.item()))

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        print('FINISH.')
