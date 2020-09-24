# BlueBird

Simple deep learning library. 

## Usage

Here is a simple implemetation of a model in bluebird.

```
from bluebird.nn import NeuralNet
from bluebird.activations import Relu, Softmax
from bluebird.layers import Input, Dense
from bluebird.data import BatchIterator
from bluebird.loss import CategoricalCrossEntropy
from bluebird.optimizers import SGD

# create the neural net
net = NeuralNet([
    Input(200), # input layer
    Dense(100, activation=Relu()),  # hidden layers with relu activation
    Dense(50, activation=Relu()),
    Dense(10, activation=Softmax()) # last hiddent layer with softmax activation
])

# define optimizer and loss function
net.build(optimizer=SGD(lr=0.003), loss=CategoricalCrossEntropy())

# train your model
net.fit(X_train, y_train, num_epochs=20)
```

## Roadmap

There are a lot of updates planed, you will find comments throughout the library that define what features I'm planing to add in the future.

## Contribution

Feel free to help, I know that there are many things that need to be optimized and implemented in the future, any help is welcome.