from layers.layers import Dense
from operations.main_operations import Linear, Sigmoid
from models.neunet import NeuralNetwork
from losses.losses import MeanSquaredError
from trainers.trainers import Trainer
from optimizers.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from funcs import to_2d_np, eval_regression_model
import pandas as pd


# put one seed in all models
SEED = 20220213

# creating different models to be compared
linear_regression = NeuralNetwork(
    layers=[Dense(neurons=1,
                  activation=Linear())
            ],
    loss=MeanSquaredError(),
    seed=SEED
)

simple_nn = NeuralNetwork(
    layers=[Dense(neurons=13,
                  activation=Sigmoid()),
            Dense(neurons=1,
                  activation=Linear())
            ],
    loss=MeanSquaredError(),
    seed=SEED
)

deep_learning = NeuralNetwork(
    layers=[Dense(neurons=13,
                  activation=Sigmoid()),
            Dense(neurons=13,
                  activation=Sigmoid()),
            Dense(neurons=1,
                  activation=Linear())
            ],
    loss=MeanSquaredError(),
    seed=SEED
)

# loading boston data
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# normalizing data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# separating
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size=0.3,
                                                    random_state=80718
                                                    )

# reshaping
y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)

#
trainer_lr = Trainer(linear_regression, SGD(lr=0.01))
trainer_lr.fit(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=50,
    seed=SEED
)
print()
eval_regression_model(linear_regression,
                      X_test,
                      y_test)
print('---------------------------------------------')

#
trainer_nn = Trainer(simple_nn, SGD(lr=0.01))
trainer_nn.fit(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=50,
    seed=SEED
)
print()
eval_regression_model(simple_nn,
                      X_test,
                      y_test)
print('---------------------------------------------')

#
trainer_dl = Trainer(deep_learning, SGD(lr=0.01))
trainer_dl.fit(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=50,
    seed=SEED
)
print()
eval_regression_model(deep_learning,
                      X_test,
                      y_test)

