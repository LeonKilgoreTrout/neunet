from numpy import ndarray
from funcs import assert_same_shape


class Operation(object):
    """
    Base class operation in our neural network

    input: input tensor (e.g. X);
    output: input tensor transformed to output by some function in forward pass (e.g function(X));
    input_grad: is output_grad for the next step in backward pass (e.g function_derivative())

    """
    input_: ndarray
    output: ndarray
    input_grad: ndarray

    def __init__(self):

        pass

    def forward(self, input_: ndarray):

        self.input_ = input_
        self.output = self._output()

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:

        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _output(self) -> ndarray:

        raise NotImplementedError

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        raise NotImplementedError


class ParamOperation(Operation):
    """
    Base class operation in our neural network

    param_grad:

    """
    param_grad: ndarray

    def __init__(self, param: ndarray):

        super(Operation).__init__()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:

        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)
        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:

        raise NotImplementedError





