#!/usr/bin/env python3
import numpy as np
BidirectionalCell = __import__('5-bi_forward').BidirectionalCell

np.random.seed(5)
cell = BidirectionalCell(10, 15, 5)
print("Whf:", cell.Whf); print("Whb:", cell.Whb); print("Wy:", cell.Wy)
print("bhf:", cell.bhf); print("bhb:", cell.bhb); print("by:", cell.by)
cell.bhf = np.random.randn(1, 15)
h_prev   = np.random.randn(8, 15)
x_t      = np.random.randn(8, 10)
h = cell.forward(h_prev, x_t)
print(h.shape); print(h)
