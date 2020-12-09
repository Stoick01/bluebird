"""
Adam optimizer
==============

Like Adaptive gradient, Adam (Adaptive Moment Estimator), also has adaptive learning rate.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nn import NeuralNet

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Input
from bluebird.activations import Activation

from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer.

    Example::

        optim = Adam(lr=0.005)
        net.build(optimizer=optim)
    """

    def __init__(self,
                 lr: float = 0.001,
                 b1: float = 0.9,
                 b2: float = 0.999,
                 t: int = 0,
                 epsilon: float = 1e-8) -> None:
        """
        Initializes the object.

        Args:
            lr (float, optional): learning rate, defaults to 0.001
            b1 (float, optional): used to decay the running average of the gradient, defaults to 0.9
            b2 (float, optional): used to decay the running average of the squared gradient, defaults to 0.999
            t (int, otpional): time step, best to leave at zero, defaults to 0
            epsilon (float, optional): small value to scape the division by zero, best to leave it alone, defaults to 1e-8
        """
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.mn = None
        self.vn = None
        self.t = t
        self.epsilon = epsilon

    def build(self, net: 'NeuralNet') -> None:
        """
        Called before training, optimizer needs the model to be able to iterate over params.

        This function is called douring build in your model.

        Args:
            net (:obj:`NeuralNet`): your model

        """

        self.net = net

        if self.mn == None and self.vn == None:
            self.mn = []
            self.vn = []
            for layer in self.net.get_layers():
                if isinstance(layer, Input) or isinstance(layer, Activation):
                    continue
                self.mn.append(np.zeros((layer.input_size, layer.output_size)))
                self.mn.append(np.zeros(layer.output_size))
                self.vn.append(np.zeros((layer.input_size, layer.output_size)))
                self.vn.append(np.zeros(layer.output_size))

    def step(self) -> None:
        """
        Traning step.

        This function is run during each of your training steps, it updates the model
        
        """

        self.t += 1
              
        for ((param, grad), m, v) in zip(self.net.get_params_and_grads(), self.mn, self.vn):
            m = self.b1 * m + (1 - self.b1) * grad
            v = self.b2 * v + (1 - self.b2) * (grad ** 2)
            mt = m / (1 - np.power(self.b1, self.t))
            vt = v / (1 - np.power(self.b2, self.t))
            param -= self.lr * mt / (np.sqrt(vt) + self.epsilon)