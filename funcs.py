import numpy as np
from numpy import ndarray
from typing import Tuple


def assert_same_shape(array: ndarray,
                      array_grad: ndarray) -> None:

    assert array.shape == array_grad.shape, \
        '''
            Two ndarrays should have the same shape. Instead, 
            first ndarray's shape is {0} and second ndarray's shape is {1}.
        '''.format(tuple(array.shape), tuple(array_grad.shape))


def permute_data(X: ndarray,
                 y: ndarray) -> Tuple[ndarray, ndarray]:

    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


# mean absolute error
def mae(y_true: ndarray,
        y_pred
        ) -> float:

    assert_same_shape(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))


# root mean squared_error
def rmse(y_true: ndarray,
         y_pred: ndarray,
         ) -> float:

    assert_same_shape(y_pred, y_true)
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))
