from typing import List
from layers.layers import Layer
from losses.losses import Loss
from numpy import ndarray


class NeuralNetwork(object):
    """
    NN
    """
    def __init__(self,
                 layers: List[Layer],
                 loss: Loss,
                 seed: float = 1
                 ):

        self.layers = layers
        self.loss = loss
        self.seed = seed
        if self.seed:
            for layer in self.layers:

                setattr(layer, 'seed', self.seed)

    def forward(self,
                x_batch: ndarray) -> ndarray:

        x_out = x_batch
        for layer in self.layers:

            x_out = layer.forward(x_out)

        return x_out

    def backward(self,
                 loss_grad: ndarray) -> None:

        grad = loss_grad
        for layer in reversed(self.layers):

            grad = layer.backward(grad)

        return

    def train_batch(self,
                    x_batch: ndarray,
                    y_batch: ndarray) -> float:

        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss

    def params(self):

        for layer in self.layers:
            yield from layer.params

    def params_grad(self):

        for layer in self.layers:
            yield from layer.param_grads
