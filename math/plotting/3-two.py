#!/usr/bin/env python3
"""
Module pour visualiser la décroissance radioactive comparative
du C-14 et du Ra-226
"""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Trace deux courbes de décroissance radioactive :
    - C-14 (demi-vie : 5730 ans)
    - Ra-226 (demi-vie : 1600 ans)
    Les courbes sont tracées avec des styles différents pour
    une meilleure lisibilité.
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730  # Demi-vie du C-14
    t2 = 1600  # Demi-vie du Ra-226
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y1, '--r', label='C-14')
    plt.plot(x, y2, '-g', label='Ra-226')

    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of Radioactive Elements')

    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    plt.legend(loc='upper right')
    plt.show()
