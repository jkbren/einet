"""
utilities.py
--------------------
Plotting and other utility code related to the paper:

Klein, B. & Hoel, E. (2019)
Uncertainty and causal emergence in complex networks.

author: Brennan Klein
email: brennanjamesklein at gmail dot com
"""

import matplotlib.pyplot as plt
import numpy as np


def show_values(pc, ax, fontsize=16, fmt="%.3f", **kw):
    """
    For bar charts, show the value of the height of the bar.
    """
    pc.update_scalarmappable()
    for p, color, value in zip(
            pc.get_paths(), pc.get_facecolors(), pc.get_array()):

        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center",
                size=fontsize, color=color, **kw)


def softmax(A, k=1.0):
    """
    Calculates the softmax of a distribution, modulated by a precision term, k.

    Parameters
    ----------
    A (np.ndarray): a vector of real-valued numbers
    k (float): a factor that modulates how precise the output softmaxed vector
               will end up being (k=1.0 is standard, k=0.0 makes it uniform).

    Returns
    -------
    A (np.ndarray): a softmaxed version of the original vector

    """
    A = np.array(A) if not isinstance(A, np.ndarray) else A
    A = A*k
    maxA = A.max()
    A = A - maxA
    A = np.exp(A)
    A = A/np.sum(A)

    return A
