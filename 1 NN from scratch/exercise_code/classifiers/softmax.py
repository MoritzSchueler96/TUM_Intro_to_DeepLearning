"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N, containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    import math

    [N,D] = np.shape(X)
    [D,num_classes] = np.shape(W)

    nll = 0.0
    reg_loss = 0.0

    denom = np.zeros(N)
    grad_temp = np.zeros((N,num_classes))

    # calculate Matrix product: X@W
    XW = np.zeros((N,num_classes))

    for i in range(N):
        for j in range(num_classes):
            for k in range(D):
                XW[i,j] = XW[i,j] + X[i,k] * W[k,j]
            denom[i] = denom[i] + math.exp(XW[i,j])

    # calculate nll and gradient_temp
    for i in range(N):
        # hot encode y
        y_he = np.zeros(10)
        y_he[y[i]] = 1

        for c in range(num_classes):
            nom = np.exp(XW[i,c])
            soft = nom / denom[i]

            nll = nll - y_he[c] * np.log(soft)
            grad_temp[i,c] = (soft - y_he[c])

    sum1 = 0.0
    # calculate reg_loss
    for c in range(num_classes):
        for d in range(D):
            sum1 = sum1 + W[d, c] * W[d, c]
        reg_loss = reg_loss + 0.5 * reg * np.sqrt(sum1)

    # calculate gradient
    grad = np.zeros((D,num_classes))

    for i in range(D):
        for j in range(num_classes):
            for k in range(N):
                grad[i,j] = grad[i,j] + X.T[i,k] * grad_temp[k,j]

    # calculate loss and gradient with regularization
    loss = nll / N + reg_loss
    dW = grad / N + reg * W

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def softmax_unstable(phi):
    """
    Calls the softmax
    
    :param phi: A numpy array of shape (N, C) containing the vector product of X@W.
    :return soft: A numpy array of shape (N, C) containing the predicted classes
    """
    soft = np.exp(phi) / np.sum(np.exp(phi))

    return soft


def softmax(phi):
    """
    Calls the softmax

    :param phi: A numpy array of shape (N, C) containing the vector product of X@W.
    :return soft: A numpy array of shape (N, C) containing the predicted classes
    """

    soft0 = np.exp(phi) / np.sum(np.exp(phi), axis=1, keepdims=True)

    phi_max= np.amax(phi, axis=1, keepdims=True)
    num = np.exp(phi - phi_max)
    denom = np.sum(num, axis=1, keepdims=True)
    soft = num / denom

    num = np.exp(phi.T - np.max(phi, axis=-1))
    soft2 = (num / num.sum(axis=0)).T

    num = np.exp(phi - np.max(phi))
    soft3 = num / num.sum(axis=-1, keepdims=True)

    soft4 = (np.exp(phi).T / np.exp(phi).sum(axis=-1)).T

    phi -= np.max(phi, axis=-1, keepdims=True)
    soft5 = np.exp(phi) / np.exp(phi).sum(axis=-1, keepdims=True)

    return soft


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    [N,D] = np.shape(X)
    [D,num_classes] = np.shape(W)


    # one hot encoded y
    y_he = np.zeros((N, num_classes))
    y_he[range(N), y] = 1

    # calculate negative log likelihood
    soft = softmax(X@W)
    error = -(y_he * np.log(soft))
    nll = sum(sum(error))

    """
     # alternative 1:
    error = -(y_he * np.log(soft))
    nll = np.sum(error)
    # alternative 2:
    error1 = -np.log(soft[range(N), y])
    nll1 = np.sum(error1)
    """

    # calculate regularization
    reg_loss = reg * 0.5 * np.linalg.norm(W[1:])**2

    # calculate loss with regularization
    loss = nll / N + reg_loss

    # calculate Gradient
    # old not working: grad = X.T @ (soft[range(N), y] - 1)
    grad = X.T @ (soft - y_he)
    dW = grad / N + reg * W

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val, learning_rates=None, regularization_strengths=None, num_iters=None):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    # assign default values
    if learning_rates is None:
        learning_rates = [2.5e-6, 3e-6]
    if regularization_strengths is None:
        regularization_strengths = [3e2, 4e2, 4.5e2, 5e2, 6e2]
    if num_iters is None:
        num_iters = [4000]

    for lr in learning_rates:
        for reg in regularization_strengths:
            # train model
            soft = SoftmaxClassifier()
            soft.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=num_iters, verbose=True)

            # calculate train acc
            y_train_pred = soft.predict(X_train)
            TPTN = np.sum(y_train_pred == y_train)
            train_acc = TPTN / len(y_train)
            # train_acc = np.mean(y_train_pred == y_train)

            # calculate val acc
            y_val_pred = soft.predict(X_val)
            TPTN = np.sum(y_val_pred == y_val)
            val_acc = TPTN / len(y_val)
            # val_acc = np.mean(y_val_pred == y_val)

            # store data in results dictionary
            results[lr, reg] = (train_acc, val_acc)
            all_classifiers.append(soft)

            # store best values
            if val_acc > best_val:
                best_val = val_acc
                best_softmax = soft

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
