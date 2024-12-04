#!/usr/bin/env python3

def matrix_shape(matrix):
    dimensions = []
    current = matrix
    while isinstance(current, list):
        dimensions.append(len(current))
        if len(current) > 0:
            current = current[0]
        else:
            break
    return dimensions
