from copy import deepcopy
from typing import Tuple
from numpy import ndarray
import numpy as np
from funcs import permute_data
from models.neunet import NeuralNetwork
from optimizers.optimizers import Optimizer


class Trainer(object):

    def __init__(self,
                 net: NeuralNetwork,
                 optim: Optimizer
                ):

        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)

    @staticmethod
    def generate_batches(X: ndarray,
                         y: ndarray,
                         size: int = 32
                         ) -> Tuple[ndarray]:

        assert X.shape[0] == y.shape[0], \
            '''
            Ndarrays should have equal shapes. Instead, {0} and {1} shapes given. 
            '''.format(X.shape[0], y.shape[0])

        N = X.shape[0]
        for ii in range(1, N, size):

            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]
            yield X_batch, y_batch

    def fit(self,
            X_train: ndarray,
            y_train: ndarray,
            X_test: ndarray,
            y_test: ndarray,
            epochs: int = 100,
            eval_every: int = 10,
            batch_size: int = 32,
            seed: int = 1,
            restart: bool = True
            ):

        np.random.seed(seed)
        if restart:

            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        for epoch in range(epochs):

            if (epoch+1) % eval_every == 0:

                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                self.net.train_batch(X_batch, y_batch)
                self.optim.step()

            if (epoch+1) % eval_every == 0:

                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)

                if loss < self.best_loss:

                    print(f'Validation loss after {epoch+1} epocs is {loss:.3f}')
                    self.best_loss = loss

                else:

                    print(f'Loss increased after epoch {epoch+1}. Final loss was {self.best_loss:.3f}')
                    self.net = last_model
                    setattr(self.optim, 'net', self.net)
                    break
