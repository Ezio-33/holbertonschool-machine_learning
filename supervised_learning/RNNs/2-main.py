#!/usr/bin/env python3
import numpy as np
GRUCell = __import__('2-gru_cell').GRUCell

np.random.seed(2)
cell = GRUCell(10, 15, 5)
print("Wz:", cell.Wz); print("Wr:", cell.Wr)
print("Wh:", cell.Wh); print("Wy:", cell.Wy)
print("bz:", cell.bz); print("br:", cell.br)
print("bh:", cell.bh); print("by:", cell.by)
cell.bz, cell.br, cell.bh, cell.by = (
    np.random.randn(1, 15),
    np.random.randn(1, 15),
    np.random.randn(1, 15),
    np.random.randn(1, 5),
)
h_prev = np.random.randn(8, 15)
x_t    = np.random.randn(8, 10)
h, y   = cell.forward(h_prev, x_t)
print(h.shape); print(h)
print(y.shape); print(y)
