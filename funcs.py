import numpy as np
from numpy import ndarray


def assert_same_shape(array: ndarray,
                      array_grad: ndarray) -> None:

    assert array.shape == array_grad.shape, \
        '''
            Two ndarrays should have the same shape. Instead, 
            first ndarray's shape is {0} and second ndarray's shape is {1}.
        '''.format(tuple(array.shape), tuple(array_grad.shape))
