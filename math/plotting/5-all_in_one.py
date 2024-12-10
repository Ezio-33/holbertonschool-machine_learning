#!/usr/bin/env python3
"""
Module pour créer une visualisation combinée
de différents types de graphiques
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    Crée une figure contenant 5 graphiques différents
    avec un espacement optimal
    """
    # Configuration initiale de la figure
    plt.figure(figsize=(6.4, 4.8))
    plt.suptitle('All in One', fontsize='x-large')

    # Graphique 1
    plt.subplot(3, 2, 1)
    y0 = np.arange(0, 11) ** 3
    plt.plot(np.arange(0, 11), y0, 'r-')
    plt.yticks([0, 500, 1000])
    plt.xticks([0, 2, 4, 6, 8, 10])
    plt.tick_params(axis='both', labelsize='x-small')
    plt.xlim(0, 10)

    # Graphique 2
    plt.subplot(3, 2, 2)
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180
    plt.scatter(x1, y1, c='magenta', s=1)
    plt.yticks([170, 180, 190])
    plt.xticks([60, 70, 80])
    plt.xlabel('Height (in)', fontsize='x-small')
    plt.ylabel('Weight (lbs)', fontsize='x-small')
    plt.title("Men's Height vs Weight", fontsize='x-small')

    # Graphique 3
    plt.subplot(3, 2, 3)
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)
    plt.plot(x2, y2)
    plt.xticks([0, 10000, 20000])
    plt.yscale('log')
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.title('Exponential Decay of C-14', fontsize='x-small')
    plt.xlim(0, 28650)

    # Graphique 4
    plt.subplot(3, 2, 4)
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31, t32 = 5730, 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)
    plt.plot(x3, y31, '--r', label='C-14')
    plt.plot(x3, y32, '-g', label='Ra-226')
    plt.yticks([0.0, 0.5, 1.0])
    plt.xticks([0, 5000, 10000, 15000, 20000])
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.title('Exponential Decay of Radioactive Elements', fontsize='x-small')
    plt.legend(fontsize='x-small')
    plt.axis([0, 20000, 0, 1])

    # Graphique 5
    plt.subplot(3, 2, (5, 6))
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.yticks([0, 10, 20, 30])
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xlabel('Grades', fontsize='x-small')
    plt.ylabel('Number of Students', fontsize='x-small')
    plt.title('Project A', fontsize='x-small')
    plt.xlim(0, 100)

    plt.tight_layout()
    plt.show()
