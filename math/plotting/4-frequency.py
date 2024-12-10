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

    bins = np.arange(0, 101, 10)
    plt.xlabel('Grades')
    plt.ylim(0, 30)
    plt.xlim(0, 100)
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.hist(student_grades, bins, edgecolor='black')
    plt.xticks(np.arange(0, 110, 10))
    plt.show()
