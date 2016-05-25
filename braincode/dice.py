from __future__ import division
import numpy as np


def dice_coefficient(A, B):
    """find dice distance between sets of ints A and B

    A and B are each a sequence of ints:
    >>> A = {1, 3, 4, 9}
    >>> B = {3, 5, 9}
    >>> result = dice_coefficient( A, B )
    >>> result == 4.0/7.0
    True

    """
    overlap = len(A & B)
    return overlap * 2.0 / (len(A) + len(B))


def dicesim_from_dot(X, nan_to_one=True):
    """Computes the dice similarity matrix using dot.
    Values in X must be only either 1.0 or 0.0 for this to be valid

    >>> X = np.array([[0, 1, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]])
    >>> D = dicesim_from_dot(X, nan_to_one=False)
    >>> expectedD = np.array([[1, 0.8, 0], [0.8, 1, 0], [0, 0, np.nan]])
    >>> np.testing.assert_array_equal(D, expectedD)
    >>> D = dicesim_from_dot(X, nan_to_one=True)
    >>> expectedD = np.array([[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1]])
    >>> np.testing.assert_array_equal(D, expectedD)
    """
    AND = X.dot(X.T)  # |A and B|; if dense, let a fast blas shine
    AB = np.add.outer(X.sum(axis=1), X.sum(axis=1))  # |A| + |B|
    with np.errstate(divide='ignore', invalid='ignore'):
        DICE = 2 * AND / AB
    if nan_to_one:
        # |A| + |B| == 0 iff A = B = {} => DICE = 1
        DICE[np.isnan(DICE)] = 1
    return DICE


def dicedist(X):
    return 1 - dicesim_from_dot(X)


def dicedist_metric(X):
    # Or we could use Tanimoto / Jaccard
    # https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient#Difference_from_Jaccard
    return np.sqrt(dicedist(X))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
