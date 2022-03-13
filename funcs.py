import numpy as np
from numpy import ndarray
from typing import Tuple
from models.neunet import NeuralNetwork


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


def to_2d_np(array: ndarray,
             type_: str = 'col'):

    assert array.ndim == 1, 'Wrong tensor given. This function requires 1d array'
    if type_ == 'col':

        return array.reshape(-1, 1)

    elif type_ == 'col':

        return array.reshape(1, -1)

    else:
        if type(type_) == str:
            raise ValueError('Wrong type_ parameter given')
        else:
            raise TypeError('Wrong type of type_ parameter given ;)')


def eval_regression_model(model: NeuralNetwork,
                          X_test: ndarray,
                          y_test: ndarray):

    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print('MSE: ', mae(preds, y_test))
    print()
    print('RMSE: ', rmse(preds, y_test))
