#!/usr/bin/env python3
import numpy as np
BidirectionalCell = __import__('6-bi_backward').BidirectionalCell
bi_rnn            = __import__('8-bi_rnn').bi_rnn

np.random.seed(8)
cell = BidirectionalCell(10, 15, 5)
X    = np.random.randn(6, 8, 10)
h_0  = np.zeros((8, 15))
h_T  = np.zeros((8, 15))
H, Y = bi_rnn(cell, X, h_0, h_T)
print(H.shape); print(H)
print(Y.shape); print(Y)
