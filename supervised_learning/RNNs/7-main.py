#!/usr/bin/env python3
import numpy as np
BidirectionalCell = __import__('7-bi_output').BidirectionalCell

np.random.seed(7)
cell = BidirectionalCell(10, 15, 5)
H_f  = np.random.randn(6, 8, 15)
H_b  = np.random.randn(6, 8, 15)
Y    = cell.output(np.concatenate((H_f, H_b), axis=-1))
print(Y.shape); print(Y)
