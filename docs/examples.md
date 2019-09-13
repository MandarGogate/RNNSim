# Wine Data Set Classification using RNN
```python
from sklearn import preprocessing
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from rnnsim.model import SequentialRNN
from rnnsim.utils import one_hot_encoder

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X, y = load_wine(True)
X_norm = min_max_scaler.fit_transform(X)
Y_one_hot = one_hot_encoder(y, num_classes=3)
X_train, X_test, y_train, y_test = train_test_split(X_norm, Y_one_hot, test_size=0.33, random_state=42)
sequential_model = SequentialRNN([13, 8, 3])
sequential_model.compile()
sequential_model.fit(train_data=(X_train, y_train), epochs=50, metrics="acc")
print(sequential_model.score((X_test, y_test)))
```

# Simple Addition RNN
```python
import random

from rnnsim.RNN import RNN
import numpy as np

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
```

# The Iris Flower Dataset Classification using RNN
```
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from rnnsim.model import SequentialRNN
from rnnsim.utils import one_hot_encoder

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X, y = load_iris(True)
X_norm = min_max_scaler.fit_transform(X)
Y_one_hot = one_hot_encoder(y, num_classes=3)
X_train, X_test, y_train, y_test = train_test_split(X_norm, Y_one_hot, test_size=0.33, random_state=42)
sequential_model = SequentialRNN([4, 9, 3])
sequential_model.compile()
sequential_model.fit(train_data=(X_train, y_train), epochs=50, metrics="acc")
print(sequential_model.score((X_test, y_test)))
```