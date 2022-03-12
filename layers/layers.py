from typing import List
from operations.basic_operations import Operation, ParamOperation
from operations.main_operations import Sigmoid, WeightMultiply, BiasAdd
from numpy import ndarray
import numpy as np
from funcs import assert_same_shape


class Layer(object):
    """
    Base layer
    """
    input_: ndarray
    output: ndarray

    def __init__(self,
                 neurons: int):

        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []
        self.operations: List[Operation] = []

    def setup_layer(self, num_in: int) -> None:

        raise NotImplementedError

    def forward(self, input_: ndarray) -> ndarray:

        if self.first:

            self.setup_layer(input_)
            self.first = False

        self.input_ = input_
        for operation in self.operations:
            input_ = operation.forward(self.input_)

        self.output = input_
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:

        assert_same_shape(self.output, output_grad)
        for operation in reversed(self.operations):

            output_grad = operation.backward(output_grad)

        input_grad = output_grad
        self._param_grads()
        return input_grad

    def _param_grads(self) -> ndarray:

        self.param_grads = []
        for operation in self.operations:

            if issubclass(operation.__class__, ParamOperation):

                self.params.append(operation.param_grad)

    def _params(self) -> ndarray:

        self.params = []
        for operation in self.operations:

            if issubclass(operation.__class__, ParamOperation):

                self.params.append(operation.params)


class Dense(Layer):
    """
    Dense Layer
    """
    seed: int

    def __init__(self,
                 neurons: int,
                 activation: Operation = Sigmoid()):

        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: ndarray) -> None:

        if self.seed:
            np.random.seed(self.seed)

        self.params = []
        self.params.append(np.random.randn(input_.shape[1], self.neurons))
        self.params.append(np.random.randn(1, self.neurons))
        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation
                           ]

        return

    