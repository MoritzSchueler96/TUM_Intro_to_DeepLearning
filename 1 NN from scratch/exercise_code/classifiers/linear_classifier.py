"""Linear Classifier Base Class."""
# pylint: disable=invalid-name
import numpy as np


class LinearClassifier(object):
    """Linear Classifier Base Class."""

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            # idx berechnen
            X_batch1 = None
            idx_start = (it * batch_size) % np.shape(X)[0]
            idx_end = ((it+1) * batch_size) % (np.shape(X)[0])

            # idx = np.random.choice(np.shape(X)[0], batch_size)

            if idx_end < idx_start:
                if idx_end == 0:
                    idx_end = np.shape(X)[0]
                elif it == num_iters:
                    idx_end = np.shape(X)[0]
                else:
                    samples = idx_end
                    idx_end = np.shape(X)[0]
                    X_batch1 = X[0:samples, :]
                    y_batch1 = y[0:samples]

            X_batch = X[idx_start:idx_end, :]
            y_batch = y[idx_start:idx_end]

            if X_batch1 is not None:
                X_batch = np.concatenate((X_batch, X_batch1), axis=0)
                y_batch = np.concatenate((y_batch, y_batch1), axis=0)

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################

            self.W = self.W - (learning_rate * grad)

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and (it % 100 == 0 or it == num_iters-1):
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each row is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################

        phi = X@self.W
        soft = np.exp(phi) / np.sum(np.exp(phi), axis=1, keepdims=True)
        y_pred = np.argmax(soft, axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        raise NotImplementedError
