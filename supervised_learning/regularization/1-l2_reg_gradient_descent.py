#!/usr/bin/env python3
"""
L2 Regularization Cost.
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    return cost + model.losses