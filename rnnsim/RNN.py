import timeit
from random import uniform

import numpy as np
from numpy.linalg import inv


class RNN(object):
    def __init__(
            self,
            n_total,
            input_neurons,
            output_neurons,
            conn_plus,
            conn_minus,
            mse_threshold=0.0005,
            lr=0.1,
            epochs=20,
            r_out=0.1,
            rand_range=0.2,
            metrics="mse",
    ):
        """
        :param n_total: total number of neurons in the network
        :param input_neurons: total number of input neurons
        :param output_neurons: total number of output neurons
        :param conn_plus: connection matrix for positive weights e.g. { 1: [3,4], 2: [3,4], 3: [5], 4: [5], 5: []} neuron 1 and 2 is connected to 3 and 4, neuron 3 and 4 is connected to 5, and neuron 5 is the output neuron.
        :param conn_minus: connection matrix for negative weights e.g. { 1: [3,4], 2: [3,4], 3: [5], 4: [5], 5: []} neuron 1 and 2 is connected to 3 and 4, neuron 3 and 4 is connected to 5, and neuron 5 is the output neuron.
        :param mse_threshold: minimum mse for early stopping
        :param lr: learning rate
        :param epochs: number of epochs
        :param r_out: Firing Rate of the Output Neurons
        :param rand_range: Weights initialization in range (0, rand_range)
        :param metrics: Evaluation metric choose either acc or mse
        """
        self.kind = 1
        assert metrics in ["acc", "mse"], "metrics must be either acc or mse"
        self.n_total = n_total
        self.wplus_index = []
        self.wminus_index = []
        self.wminus_conn = []
        self.wplus_conn = []
        self.initialise_connections(conn_minus, conn_plus, n_total)
        self.n_input = len(input_neurons)
        self.n_output = len(output_neurons)
        self.output_index = np.array(output_neurons)
        self.input_index = np.array(input_neurons)
        self.mse_threshold = mse_threshold
        self.lr = lr
        self.epochs = epochs
        self.r_out = r_out
        self.rand_range = rand_range
        self.fix_r_in = 0  # DO NOT CHANGE (Related to RNN function approximation)
        self.r_in = 1  # DO NOT CHANGE (Related to RNN function approximation)
        self.auto_mapping = (
            0
        )  # Flag = 1 only if the network is recurrent with shared inputs/outputs
        self.iter = 0
        self.r = np.zeros(self.n_total)
        self.wplus = np.zeros((self.n_total, self.n_total))
        self.wminus = np.zeros((self.n_total, self.n_total))
        self.w = np.zeros((self.n_total, self.n_total))
        self.accuracy = 0
        self.metrics = metrics

    def initialise_connections(self, conn_minus, conn_plus, n_total):
        for i in range(1, n_total + 1):
            if conn_plus.get(i):
                self.wplus_conn.append(conn_plus[i])
                self.wplus_index.append(len(conn_plus[i]))
            else:
                self.wplus_conn.append([])
                self.wplus_index.append(0)
            if conn_minus.get(i):
                self.wminus_conn.append(conn_minus[i])
                self.wminus_index.append(len(conn_minus[i]))
            else:
                self.wminus_conn.append([])
                self.wminus_index.append(0)

    def initialise_weights(self):
        for i in range(self.n_total):
            for j in range(self.wplus_index[i]):
                self.wplus[i][self.wplus_conn[i][j] - 1] = uniform(0, self.rand_range)
        for i in range(self.n_total):
            for j in range(self.wminus_index[i]):
                self.wminus[i][self.wminus_conn[i][j] - 1] = uniform(0, self.rand_range)

    def prepare_data(self, inputs, input_only=False):
        inp, out = (np.array(inputs[0]), np.array(inputs[1]))

        self.inp = inp
        self.datasamples = inp.shape[0]
        self.b_lambda = np.zeros((self.datasamples, self.n_input))
        self.s_lambda = np.zeros((self.datasamples, self.n_input))
        self.b_lambda = np.multiply(self.inp, self.inp >= 0.0)
        self.s_lambda = -1 * np.multiply(self.inp, self.inp < 0.0)
        self.s_lambda = np.pad(
            self.s_lambda, [(0, 0), (0, self.n_total - self.n_input)], mode='constant'
        )
        self.b_lambda = np.pad(
            self.b_lambda, [(0, 0), (0, self.n_total - self.n_input)], mode='constant'
        )
        self.q = np.zeros(self.n_total)
        self.N = np.zeros(self.n_total)
        self.D = np.zeros(self.n_total)
        if not input_only:
            self.error = []
            self.out = out
            self.out = np.pad(
                self.out, [(0, 0), (self.n_total - self.n_output, 0)], mode='constant'
            )
            self.winv = None

    def predict(self, inp):
        self.prepare_data((inp, None), input_only=True)
        return np.concatenate(
            tuple(
                [
                    self.forward(index)[self.output_index - 1][np.newaxis, ...]
                    for index in range(inp.shape[0])
                ]
            )
        )

    def save_weights(self, path):
        np.savez_compressed(path, wplus=self.wplus, wminus=self.wminus)

    def load_weights(self, path):
        data = np.load(path)
        self.wplus = data["wplus"]
        self.wminus = data["wminus"]

    def calculate_rate(self):
        self.r = np.zeros(self.n_total)
        assert self.fix_r_in == 0
        self.r = self.r + np.sum(self.wplus, axis=1) + np.sum(self.wminus, axis=1)
        if self.auto_mapping == 0:
            self.r[self.output_index - 1] = self.r_out

    def forward(self, k):
        q = np.zeros(self.n_total)
        for i in range(self.n_total):
            self.N[i] = 0.0
            self.D[i] = 0.0
            for j in range(self.n_total):
                self.N[i] += q[j] * self.wplus[j][i]
                self.D[i] += q[j] * self.wminus[j][i]
            self.N[i] += self.b_lambda[k][i]
            self.D[i] += self.r[i] + self.s_lambda[k][i]
            if self.D[i] != 0.0:
                q[i] = self.N[i] / self.D[i]
            if self.D[i] == 0.0:
                q[i] = 1.0
            if q[i] > 1.0:
                q[i] = 1.0
            if q[i] < 0.0:
                q[i] = 0.0
        return q

    def calculate_loss(self, data_index):
        mse = 0.0
        a = self.q[self.output_index - 1] - self.out[data_index][self.output_index - 1]
        mse = mse + np.sum(np.multiply(a, a))
        if self.metrics == "acc":
            if np.argmax(self.q[self.output_index - 1]) == np.argmax(
                    self.out[data_index][self.output_index - 1]
            ):
                self.accuracy += 1
        self.mse_avg += mse

    def backprop(self):
        for i in range(self.n_total):
            self.w[:, i] = (
                                   self.wplus[:, i] - (self.wminus[:, i] * self.q[i])
                           ) / self.D[i]
        return inv(np.identity(self.n_total) - self.w)

    def update_wplus(self, k):
        wplus_result = np.zeros((self.n_total, self.n_total))
        gammaplus = np.zeros(self.n_total)
        for u in range(self.n_total):
            for vv in range(self.wplus_index[u]):
                v = self.wplus_conn[u][vv] - 1
                for i in range(self.n_total):
                    if (u != i) and (v == i):
                        gammaplus[i] = 1.0 / self.D[i]
                    elif (u == i) and ((v - 1) != i):
                        gammaplus[i] = -1.0 / self.D[i]
                avg_loss = 0
                for jj in range(self.n_output):
                    i = self.output_index[jj] - 1
                    vmplus = (
                            gammaplus[u] * self.winv[u][i] + gammaplus[v] * self.winv[v][i]
                    )
                    avg_loss = (
                            avg_loss + vmplus * (self.q[i] - self.out[k][i]) * self.q[u]
                    )
                wplus_result[u][v] = self.wplus[u][v] - self.lr * avg_loss
        wplus_result = np.multiply(wplus_result, (wplus_result >= 0.0))
        return wplus_result

    def update_wminus(self, k):
        wminus_result = np.zeros((self.n_total, self.n_total))
        gammaminus = np.zeros(self.n_total)
        for u in range(self.n_total):
            for vv in range(self.wminus_index[u]):
                v = self.wminus_conn[u][vv] - 1
                for i in range(self.n_total):
                    if (u != i) and (v == i):
                        gammaminus[i] = -self.q[i] / self.D[i]
                    elif (u == i) and ((v - 1) != i):
                        gammaminus[i] = -1.0 / self.D[i]
                    elif (u == i) and ((v - 1) == i):
                        gammaminus[i] = -(1.0 + self.q[i]) / self.D[i]
                avg_loss = 0
                for jj in range(self.n_output):
                    i = self.output_index[jj] - 1
                    vmminus = (
                            gammaminus[u] * self.winv[u][i]
                            + gammaminus[v] * self.winv[v][i]
                    )
                    avg_loss = (
                            avg_loss + vmminus * (self.q[i] - self.out[k][i]) * self.q[u]
                    )
                wminus_result[u][v] = self.wminus[u][v] - self.lr * avg_loss
                if wminus_result[u][v] < 0:
                    wminus_result[u][v] = 0.0
        return wminus_result

    def fit(
            self,
            train_data,
            epochs,
            validation_data=None,
            save_weights_path=None,
            pretrained_weights_path=None,
    ):
        """
        Abstract fit function
        :param train_data: (X_train, y_train)
        :param epochs: number of epochs
        :param validation_data: (X_val, y_val)
        :param save_weights_path: paths to save weights
        :param pretrained_weights_path: load weights from pretrained weights path
        :return:
        """
        self.prepare_data(train_data)
        if pretrained_weights_path:
            self.load_weights(pretrained_weights_path)
        else:
            self.initialise_weights()
        assert self.kind == 1, "Only feedforward networks are currently supported"
        for i in range(epochs):
            start_time = timeit.default_timer()
            self.iter += 1
            self.mse_avg = 0.0
            self.accuracy = 0
            for index in range(self.datasamples):
                self.calculate_rate()
                self.q = self.forward(index)
                self.calculate_loss(index)
                self.winv = self.backprop()
                self.update_weights(index)
            self.mse_avg = self.mse_avg / self.datasamples
            self.error.append(self.mse_avg)
            if self.metrics == "acc":
                self.accuracy = self.accuracy / self.datasamples
                train_metric = "Train Accuracy: {:02.3f} %".format(self.accuracy * 100)
            else:
                train_metric = "Train Error: {:02.5f}".format(self.mse_avg)
            if validation_data:
                self.prepare_data(validation_data)
                val_metric = "Validation " + self.score(validation_data)
                self.prepare_data(train_data)
            else:
                val_metric = ""
            print(
                (
                    "Iter: {:d} {} {} Time: {:f}".format(
                        self.iter,
                        train_metric,
                        val_metric,
                        timeit.default_timer() - start_time,
                    )
                )
            )

            if self.mse_avg < self.mse_threshold:
                print("Stopped Training: Minimum Error Reached")
                break
        if save_weights_path:
            self.save_weights(save_weights_path)

    def score(self, validation_data):
        predictions = self.predict(validation_data[0])
        if self.metrics == "acc":
            acc = (
                    100
                    * np.sum(
                np.argmax(predictions, axis=-1)
                == np.argmax(validation_data[1], axis=-1)
            )
                    / predictions.shape[0]
            )
            metric = "Accuracy: {:02.3f} %".format(acc)
        else:
            error = predictions.ravel() - validation_data[1].ravel()
            mse_avg = np.sum(np.multiply(error, error)) / error.shape[0]
            metric = "Error: {:02.3f} %".format(mse_avg)
        return metric

    def update_weights(self, k):
        self.wplus = self.update_wplus(k)
        self.wminus = self.update_wminus(k)
