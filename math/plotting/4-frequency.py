#!/usr/bin/env python3
"""
Module pour créer un histogramme représentant la distribution
des notes d'étudiants pour un projet
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Crée un histogramme montrant la distribution des notes d'étudiants.
    Les notes suivent une distribution normale avec une moyenne de 68
    et un écart-type de 15.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='k')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.ylim(0, 30)
    plt.yticks([0, 10, 20, 30])
    plt.xlim(0, 100)
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.show()
