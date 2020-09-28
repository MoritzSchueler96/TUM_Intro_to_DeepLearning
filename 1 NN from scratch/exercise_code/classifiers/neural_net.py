"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np


def softmax(phi):

    try:
        soft = np.exp(phi) / np.sum(np.exp(phi), axis=1, keepdims=True)
    except:
        soft = np.exp(phi) / np.sum(np.exp(phi))
    return soft

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape

        # Compute the forward pass
        scores = None
        ########################################################################
        # TODO: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #         
        ########################################################################

        scores1 = X@W1 + b1
        scores1_relu = np.maximum(0, scores1)
        scores2 = scores1_relu @ W2 + b2

        scores = scores2

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # TODO: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################

        soft = softmax(scores)
        error = -np.log(soft[range(N), y])
        nll = np.sum(error)

        reg_loss = reg * 0.5 * (np.sum(W1 * W1) + np.sum(W2 * W2))

        loss = nll / N + reg_loss

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # TODO: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################

        # one hot encode y
        [N, D] = np.shape(X)
        [D, num_classes] = np.shape(W2)
        y_he = np.zeros((N, num_classes))  # np.zeros_like(soft)
        y_he[range(N), y] = 1  # y_he[np.arrange(N), y] = 1

        # calculate gradient of output layer
        # derivative of softmax
        dloss_dsoft = ((soft - y_he) / N)

        # derivative of output layer w.r.t. W2
        dloss_dW2 = scores1_relu.T @ dloss_dsoft
        dloss_dW2 += reg * W2

        # derivative of output layer w.r.t. b2
        dloss_db2 = dloss_dsoft.T @ np.ones(N)

        # calculate gradient of hidden layer
        # derivative of relu
        dmax = np.zeros_like(scores1)
        dmax[scores1 > 0] = 1

        # derivative of first layer w.r.t. W1
        dloss_dscores1 = (dloss_dsoft @ W2.T) * dmax
        dloss_dW1 = X.T @ dloss_dscores1
        dloss_dW1 += reg * W1

        # derivative of first layer w.r.t. b1
        dloss_db1 = dloss_dscores1.T @ np.ones(N)

        # write values to dictionary
        grads['W2'] = dloss_dW2
        grads['b2'] = dloss_db2
        grads['W1'] = dloss_dW1
        grads['b1'] = dloss_db1

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=None, learning_rate_decay=None,
              reg=None, num_iters=None,
              batch_size=None, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """

        # assign default values
        if learning_rate is None:
            learning_rate = 1e-3
        if learning_rate_decay is None:
            learning_rate_decay = 0.95
        if reg is None:
            reg = 1e-5
        if num_iters is None:
            num_iters = 100
        if batch_size is None:
            batch_size = 200

        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        all_idx = np.linspace(0, np.shape(X)[0]-1, num=np.shape(X)[0], dtype=int)
        sample_idx = all_idx
        mask = np.ones(len(all_idx), dtype=bool)

        if np.shape(all_idx)[0] < batch_size:
            batch_size = len(all_idx)

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # TODO: Create a random minibatch of training data and labels,     #
            # storing them in X_batch and y_batch respectively.                #
            ####################################################################

            if np.shape(sample_idx)[0] < batch_size:
                sample_idx = all_idx
                mask = np.ones(len(all_idx), dtype=bool)

            idx = np.random.choice(sample_idx, batch_size, replace=False)
            mask[idx] = False
            sample_idx = all_idx[mask]

            X_batch = X[idx, :]
            y_batch = y[idx]

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ####################################################################
            # TODO: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################

            self.params['W2'] = self.params['W2'] - learning_rate * grads['W2']
            self.params['b2'] = self.params['b2'] - learning_rate * grads['b2']
            self.params['W1'] = self.params['W1'] - learning_rate * grads['W1']
            self.params['b1'] = self.params['b1'] - learning_rate * grads['b1']

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################

        scores = self.loss(X)
        soft = softmax(scores)
        y_pred = np.argmax(soft, axis=1)

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred


def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val, hidden_size=None, learning_rates=None, learning_rate_decays=None, regularization_strengths=None, num_iters=None, batch_size=None):
    best_net = None # store the best model into this 

    ############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best     #
    # trained model in best_net.                                               #
    #                                                                          #
    # To help debug your network, it may help to use visualizations similar to #
    # the  ones we used above in the Jupyther Notebook; these visualizations   #
    # will have significant qualitative differences from the ones we saw for   #
    # the poorly tuned network.                                                #
    #                                                                          #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful#
    # to  write code to sweep through possible combinations of hyperparameters #
    # automatically like we did on the previous exercises.                     #
    ############################################################################

    best_val = -1
    results = {}
    all_nets = []

    # assign default values
    if hidden_size is None:
        hidden_size = [1200, 1500, 2000, 2500]
    if learning_rates is None:
        learning_rates = [1e-5, 1e-4, 1e-3]
    if learning_rate_decays is None:
        learning_rate_decays = [0.95, 0.9]
    if regularization_strengths is None:
        regularization_strengths = [1e-12, 1e-9, 1e-6, 1e-3]
    if num_iters is None:
        num_iters = 1000
    if batch_size is None:
        batch_size = 200

    input_size = np.shape(X_train)[1]
    num_classes = max(np.amax(y_train), np.amax(y_val)) + 1

    for it in num_iters:
        print('Actual Num_iters %e' % it)
        for hds in hidden_size:
            print('Number of hidden layers %e' % hds)
            for lrd in learning_rate_decays:
                print('Actual lr decay %e' % lrd)
                for lr in learning_rates:
                    print('Actual learning rate %e' % lr)
                    for reg in regularization_strengths:
                        print('Actual reg strength %e' % reg)
                        net = TwoLayerNet(input_size, hds, num_classes)
                        stats = net.train(X_train, y_train, X_val, y_val, learning_rate=lr, learning_rate_decay=lrd, reg=reg, num_iters=it, batch_size=batch_size, verbose=True)

                        # accuracy based on batch
                        train_acc = np.mean(stats['train_acc_history'])
                        val_acc = np.mean(stats['val_acc_history'])

                        # accuracy based on all train data
                        # train_acc = (net.predict(X_train) == y_train).mean()
                        # val_acc = (net.predict(X_val) == y_val).mean()

                        results[it, hds, lr, lrd, reg] = (train_acc, val_acc)
                        all_nets.append(net)

                        if val_acc > best_val:
                            best_val = val_acc
                            best_net = net

    # Print out results.
    for (it, hds, lr, lrd, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(it, hds, lr, lrd, reg)]
        print('num_iters %e hds %e lr %e lrd %e reg %e train accuracy: %f val accuracy: %f' % (
              it, hds, lr, lrd, reg, train_accuracy, val_accuracy))

    print('best validation accuracy achieved during validation: %f' % best_val)

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return best_net, results, all_nets
