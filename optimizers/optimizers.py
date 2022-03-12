from models.neunet import NeuralNetwork


class Optimizer(object):

    def __init__(self,
                 lr: float = 0.01):

        self.lr = lr

    def step(self) -> None:

        pass


class SGD(Optimizer):
    """
    SGD
    """
    net: NeuralNetwork

    def __init__(self, lr):

        super(SGD).__init__(lr)

    def step(self):

        for (param, param_grad) in zip(self.net.params(),
                                       self.net.params_grad()):

            param -= self.lr * param_grad
