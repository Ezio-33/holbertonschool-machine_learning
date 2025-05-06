#!/usr/bin/env python3
import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell
rnn     = __import__('1-rnn').rnn

np.random.seed(1)
cell = RNNCell(10, 15, 5)
cell.bh = np.random.randn(1, 15)
cell.by = np.random.randn(1, 5)
X   = np.random.randn(6, 8, 10)   # (t, m, i)
h_0 = np.zeros((8, 15))           # (m, h)
H, Y = rnn(cell, X, h_0)
print(H.shape); print(H)
print(Y.shape); print(Y)
