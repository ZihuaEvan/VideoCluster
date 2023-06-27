import numpy as np
from sklearn import metrics


def cal_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = np.array(y_true,dtype=np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def nmi(y_pred, y_true, average_method='max'):
    """
    Calculate the normalized mutual information.
    :param y_pred: Predicted labels of samples, do not need to map to the ground truth label.
                   ndarray, shape: [num_samples, ].
    :param y_true: Ground truth labels of samples. ndarray, shape: [num_samples, ].
    :param average_method: How to compute the normalizer in the denominator. Possible options
                           are 'min', 'geometric', 'arithmetic', and 'max'.
                           'min': min(U, V)
                           'geometric': np.sqrt(U * V)
                           'arithmetic': np.mean([U, V])
                           'max': max(U, V)
    :return: Normalized mutual information.
    """
    return metrics.normalized_mutual_info_score(y_true, y_pred, average_method=average_method)


def ari(y_pred, y_true):
    """
    Calculate the adjusted rand index.
    :param y_pred: Predicted labels of samples, do not need to map to the ground truth label.
                   ndarray, shape: [num_samples, ].
    :param y_true: Ground truth labels of samples. ndarray, shape: [num_samples, ].
    :return: Adjusted rand index.
    """
    return metrics.adjusted_rand_score(y_true, y_pred)