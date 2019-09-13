"""A sample module."""

import numpy as np

from .RNN import RNN


class SequentialRNN:
    """Linear stack of fully connected layers.
        # Arguments
            layers: list of layers to add to the model.
        # Example
        ```python
        sequential_model = SequentialRNN([13, 8, 3])
        # The same model can be created using
        model = SequentialRNN()
        model.add(13)
        model.add(8)
        model.add(3)
        ```
    """

    def __init__(self, layers=None):
        if layers is None:
            layers = []
        self.layers: list[int] = layers
        self.rnn = None

    def add(self, layer_size: int):
        """
        Adds a layer instance on top of the layer stack

        :param layer_size:
        """
        self.layers.append(layer_size)

    def compile(self):
        """
        Build the connection used by RNN model as per the self.layers
        """
        self.n_total = np.sum(self.layers)
        self.input_neurons = []
        self.output_neurons = []
        self.conn_plus = {}
        self.conn_minus = {}
        for i in range(self.layers[0]):
            self.input_neurons.append(i + 1)
        for i in range(self.n_total - self.layers[-1] + 1, self.n_total + 1):
            self.output_neurons.append(i)
        start = 1
        initial = 1
        for i in range(len(self.layers)):
            start += self.layers[i]
            if i == len(self.layers) - 1:
                for j in range(self.n_total - self.layers[-1] + 1, self.n_total + 1):
                    self.conn_plus[j] = []
                    self.conn_minus[j] = []
            else:
                for j in range(initial, self.layers[i] + initial):
                    self.conn_plus[j] = [
                        x for x in range(start, start + self.layers[i + 1])
                    ]
                    self.conn_minus[j] = [
                        x for x in range(start, start + self.layers[i + 1])
                    ]
            initial = start
        self.rnn = RNN(
            n_total=self.n_total,
            input_neurons=self.input_neurons,
            output_neurons=self.output_neurons,
            conn_plus=self.conn_plus,
            conn_minus=self.conn_minus,
        )

    def fit(
            self,
            train_data,
            epochs,
            validation_data=None,
            mse_threshold=0.0005,
            lr=0.1,
            r_out=0.1,
            rand_range=0.2,
            metrics="mse",
            save_weights_path=None,
            pretrained_weights_path=None,
    ):
        """

        :param train_data:
        :param epochs:
        :param validation_data:
        :param mse_threshold:
        :param lr:
        :param r_out:
        :param rand_range:
        :param metrics:
        :param save_weights_path:
        :param pretrained_weights_path:
        :return:
        """
        self.rnn = RNN(
            n_total=self.n_total,
            input_neurons=self.input_neurons,
            output_neurons=self.output_neurons,
            conn_plus=self.conn_plus,
            conn_minus=self.conn_minus,
            mse_threshold=mse_threshold,
            lr=lr,
            epochs=epochs,
            r_out=r_out,
            rand_range=rand_range,
            metrics=metrics,
        )
        self.rnn.initialise_weights()
        self.rnn.fit(
            train_data=train_data,
            epochs=epochs,
            validation_data=validation_data,
            save_weights_path=save_weights_path,
            pretrained_weights_path=pretrained_weights_path,
        )

    def score(self, test_data):
        return "Test " + self.rnn.score(test_data)
