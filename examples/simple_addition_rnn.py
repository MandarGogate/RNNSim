import random

import numpy as np

from rnnsim.RNN import RNN

N_Iterations = 10
n_total = 5
input_neurons = [1, 2]
output_neurons = [5]
conn_plus = {
    1: [3, 4], 2: [3, 4],
    3: [5], 4: [5], 5: []}
conn_minus = {
    1: [3, 4], 2: [3, 4],
    3: [5], 4: [5], 5: []}
print("Generating data")
X = np.array([[random.random() / 2, random.random() / 2] for _ in range(1000)])
Y = X.sum(axis=-1)[..., np.newaxis]

model = RNN(n_total=n_total, input_neurons=input_neurons, output_neurons=output_neurons, conn_plus=conn_plus, conn_minus=conn_minus)
model.fit(epochs=N_Iterations, train_data=(X, Y))
print("Prediction for [0.1 + 0.2]", model.predict(np.array([[0.1, 0.2]])))
print("Prediction for [0.3 + 0.4]", model.predict(np.array([[0.3, 0.4]])))
