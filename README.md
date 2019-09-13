# Overview

Random Neural Network Simulator implemented in Python.

[![PyPI Version](https://img.shields.io/pypi/v/rnnsim.svg)](https://pypi.org/project/rnnsim)
[![PyPI License](https://img.shields.io/pypi/l/rnnsim.svg)](https://pypi.org/project/rnnsim)

# Setup

## Requirements

* Python 3.6+
* NumPy
* Sklearn 

## Installation

Install this library directly into an activated virtual environment:

```bash
$ pip install rnnsim
```

or add it to your [Poetry](https://poetry.eustace.io/) project:

```bash
$ poetry add rnnsim
```

# Usage

After installation, the package can either be used as:

```python

from rnnsim.model import SequentialRNN

sequential_model = SequentialRNN([2, 2, 1])
sequential_model.compile()
sequential_model.fit(train_data=(X_train, y_train), epochs=50, metrics="acc")
print(sequential_model.score((X_test, y_test)))
```

or 

```python
from rnnsim.RNN import RNN

# define model connections
conn_plus = {
    1: [3, 4], 2: [3, 4],
    3: [5], 4: [5], 5: []}
conn_minus = {
    1: [3, 4], 2: [3, 4],
    3: [5], 4: [5], 5: []}
model = RNN(n_total=5, input_neurons=2, output_neurons=1, conn_plus=conn_plus, conn_minus=conn_minus)
model.fit(epochs=N_Iterations, train_data=(X, Y))
```
