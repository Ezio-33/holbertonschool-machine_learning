#!/usr/bin/env python3
import numpy as np
LSTMCell = __import__('3-lstm_cell').LSTMCell

np.random.seed(3)
cell = LSTMCell(10, 15, 5)
for b in ("bf", "bu", "bc", "bo", "by"):
    setattr(cell, b, np.random.randn(1, 15 if b != "by" else 5))
h_prev = np.random.randn(8, 15)
c_prev = np.random.randn(8, 15)
x_t    = np.random.randn(8, 10)
h, c, y = cell.forward(h_prev, c_prev, x_t)
print(h.shape); print(h)
print(c.shape); print(c)
print(y.shape); print(y)
