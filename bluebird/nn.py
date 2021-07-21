"""
Neural Net
==========

Default class for creating feed forward neural networs
"""

# TO DO: Model from multiple models, base model class

from bluebird.dataloader.dataloaderbase import DataLoaderBase
from typing import Sequence, Iterator, Tuple

import json
import numpy as np

from .tensor import Tensor
from .layers import Layer, Input, MaxPool2D
from .loss import Loss, MSE
from .optimizers import Optimizer, SGD
from .activations import Activation
from .weight_initializers import WeightInitializer, RandomWeightInitializer
from .dataloader import DataLoaderBase
from .data import Batch

from .exceptions import TypeException
from .progress_tracker import ProgressBar

import bluebird.utils as utl

from bluebird import dataloader


class Model():
    """
    Base model class.

    All other models inherit from it.

    Example::

        class CustomModel(Model):
            def __init__(self, layers: Sequence[Layer]):
                super.__init__()

            def build(self, 
                    iterator: DataIterator = BatchIterator(),
                    loss: Loss = MSE(),
                    optimizer: Optimizer = SGD()) -> None:
                super.build()

            def step(self, batch) -> float:
                ...

                return self.loss.loss(predicted, batch.targets) 
    """


    def __init__(self, layers: Sequence[Layer]) -> None:
        """
        Initalizes the object.

        Args:
            layers (Sequence[Layer]): sequence of layers of the newtwork

        """
        if not isinstance(layers, Sequence):
            raise TypeException("layers", "Sequence[Layer]")

        self.layers = layers

    def build(self,
            loss: Loss = MSE(),
            optimizer: Optimizer = SGD()) -> None:
        """
        Used to build the model.

        Defines aditional parameters and initializes the weights.

        Args:
            loss (:obj:`Loss`, otpional): defines loss function
                Defaults to MSE()
            optimizer (:obj:`Optimizer`, optional): defines how the weights should be updated
                Defaults to SGD()

        """
        

        if not isinstance(loss, Loss):
            raise TypeException("loss", "Loss")

        if not isinstance(optimizer, Optimizer):
            raise TypeException("optimizer", "Optimizer")

        self.loss = loss
        self.optimizer = optimizer

        dimension = 0
        for layer in self.layers:
            if isinstance(layer, Activation):
                continue

            if isinstance(layer, Input):
                layer.build()
                dimension = layer.output_size
                continue

            if isinstance(layer, MaxPool2D):
                continue

            layer.build(dimension)
            dimension = layer.output_size

        self.optimizer.build(self)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward propagates through the network.

        Args:
            inputs (:obj:`Tensor`): Network input

        Returns:
            :obj:`Tensor`: Network output

        """

        for layer in self.layers:
            inputs = layer.forward(inputs, training=True)
        return inputs


    def backward(self, grad:Tensor) -> Tensor:
        """
        Backward propagates through the network.

        Args:
            inputs (:obj:`Tensor`): Gradient that you get from the loss function

        Returns:
            :obj:`Tensor`: Gradient from the first layer

        """

        for layer in self.get_layers():
            grad = utl.grad_clip(grad)
            grad = layer.backward(grad)
        return grad

    def get_layers(self) -> Iterator[Layer]:
        """
        Returns reversed layers.

        Returns:
            Iterator[:obj:`Tensor`]: List of layer objects

        """

        for layer in reversed(self.layers):
            yield layer

    def predict(self, inputs: Tensor) -> Tensor:
        """
        Used to predict values after you finished the training.

        Args:
            inputs (:obj:`Tensor`): values you wish to predict

        Returns: 
            :obj:`Tensor`: Predicted values
        
        """

        for layer in self.layers:
            inputs = layer.forward(inputs, training=False)
        return inputs

    def get_params_and_grads(self) -> Iterator[Tensor]:
        """
        Returns parameters and gradients for each layer.

        Returns:
            Iterator[:obj:`Tensor`]: List of parameter, gradient pairs

        """

        for layer in self.get_layers():
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def step(self, batch: Batch) -> float:
        """
        Step function is called during each training step.

        It calculates the loss and updates weights and biases.

        You must return loss.

        Args:
            batch (:obj:`Batch`): recives batch object, each batch object has inputs and targets

        Raises:
            NotImplementedError

        """

        raise NotImplementedError


    def fit(self, 
            loader: DataLoaderBase,
            num_epochs: int) -> None:
        """
        Used to train the model.

        Args:
            loader (:obj:`DataLoaderBase`): data loader type object used to load data for training
            num_epochs (int): number of epochs you want to train
            
        """
        

        if not isinstance(num_epochs, int):
            raise TypeException("num_epochs", "int")

        n = loader.__len__()

        epoch_loss = 0.0
        items = 0
        for epoch in range(num_epochs):
            bar = ProgressBar(n, num_epochs)

            for batch in loader():
                items += len(batch.inputs)
                epoch_loss += self.step(batch)
                bar.print_bar(items - n*epoch, epoch+1, epoch_loss/items)

    def save(self, path: str = '.', filename: str = 'model') -> None:
        """
        Save the model to a file.

        It saves every layer, and everything within params object.

        If you want your custom layer to be saved make sure, that everything you want to save is in params.

        Everything is saved as a json object as follows:

        .. code-block::

            {
                'layerName': [
                    {
                        name: 'paramName',
                        value: '0.5888 2.59874.......'
                    },
                    ....
                ]
            }

        Args:
            path (:obj:`str`, optional): location where you want your model to be saved to,
                defaults to '.' which means this directory
            name (:obj:`str`, optional): name of the file, defaults to 'model',
                note: extension is added manually

        """

        layers = {'layers': []}

        for layer in self.layers:
            name = type(layer).__name__
            params = []

            for key, value in layer.params.items():
                if isinstance(value, np.ndarray):
                    value = value.tolist()

                p = {
                    'name': key,
                    'value': value
                }

                params.append(p)

            l = {
                name: params
            }

            layers['layers'].append(l)

        if path.endswith('/'):
            path += filename + '.json'
        else:
            path += '/' + filename + '.json'

        with open(path, 'w') as f:
            json.dump(layers, f)


    def load(self, path: str) -> None:
        """
        Load the model from a file.

        Args:
            path (:obj:`str`): path to the file where the model is saved

        """
        with open(path) as f:
            data = json.load(f)
            for loaded, layer in zip(data['layers'], self.layers):
                for key, value in loaded.items():
                    for param in value:
                        layer.params[param['name']] = param['value']

    def summary(self, shape: tuple) -> None:
        """
        Prints the information about the model (Layes, shapes, params).

        Args:
            shape (tuple): shape of the input vector

        """

        params = 0

        params_size = 0.0

        inp = np.random.randn(*shape)
        idx = 1
        print()
        print('-'*60)
        print('{:>20} {:>20} {:>20}'.format('Layer(type)', 'Output Shape', 'Param #'))
        print('='*60)

        for layer in self.layers:
            inp = layer.forward(inp)
            if isinstance(layer, Activation) or isinstance(layer, Input):
                par = None
            else:
                par = layer.get_params()

            num = 0
            if par != None:
                for k, v in par.items():
                    num += v.size
                    params_size += v.nbytes

            params += num
            idx += 1

            num = "{:,}".format(num)

            print('{:>20} {:>20} {:>20}'.format(layer.__class__.__name__, str(inp.shape), num))

        print('='*60)
        m = 'Byte'

        if params_size / 1000 > 0.1:
            params_size /= 1000
            m = 'KB'

        if params_size / 1000 > 0.1:
            params_size /= 1000
            m = 'MB'

        if params_size / 1000 > 0.1:
            params_size /= 1000
            m = 'GB'

        params = "{:,}".format(params)
        params_size = "{:.2f}".format(params_size)

        print('Total params: ' + params)
        print('Params size (' + m + '): ' + params_size)
        print('-'*60)


class NeuralNet(Model):
    """
    Creates a neural network model.

    Example::
        
        net = NeuralNet([
                Flatten(input_size=(28, 28)),
                Dense(300, activation=Relu()),
                Dense(100, activation=Relu()),
                Dense(50, activation=Relu()),
                Dense(10, activation=Softmax())
            ])
        net.build(optimizer=AdaGrad(lr=0.003), loss=CategoricalCrossEntropy())

        net.fit(X_train, y_train, num_epochs=20)

        net.predict(X_test)

    """


    def step(self, batch) -> float:
        """
        Step function is called during each training step.

        It calculates the loss and updates weights and biases.

        Args:
            batch (:obj:`Batch`): recives batch object, each batch object has inputs and targets

        Returns:
            float: returns calculated loss

        """
        predicted = self.predict(batch.inputs)

        loss = self.loss.loss(predicted, batch.targets) 
        grad = self.loss.grad(predicted, batch.targets)

        self.backward(grad)
        self.optimizer.step()

        return loss

                