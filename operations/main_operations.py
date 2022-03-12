import numpy as np
from numpy import ndarray
from funcs import assert_same_shape
from basic_operations import Operation, ParamOperation


class WeightMultiply(ParamOperation):
    """
    Class for multiplying inputs with weights (param)
    """
    def __init__(self, weights: ndarray):

        super().__init__(weights)

    def _output(self) -> ndarray:
        # forward pass matrix multiplication
        return self.input_ @ self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        # grads for inputs: W.T-like
        return output_grad @ self.param.T

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        # grads for weights (params): X.T-like
        return self.input_.T @ output_grad


class BiasAdd(ParamOperation):
    """
    Class for adding biases (param)
    """
    def __init__(self, bias: ndarray):

        assert bias.shape[0] == 1, \
            f'''
                Use .reshape(1, -1) for correct bias given. Instead, it has {bias.shape} shape.
            '''
        super().__init__(bias)

    def _output(self) -> ndarray:

        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:

        param_grad = np.ones_like(self.param) * output_grad

        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


class Sigmoid(Operation):

    def __init__(self):

        super(Sigmoid, self).__init__()

    def _output(self) -> ndarray:

        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad

        return input_grad

