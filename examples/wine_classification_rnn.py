"""
Wine Data Set Classification using RNN
These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.

https://archive.ics.uci.edu/ml/datasets/wine
"""
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
