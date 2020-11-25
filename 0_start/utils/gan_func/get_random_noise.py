import numpy as np


def get_random_noise(rows, columns, random_noise_dimension):
    """

    Parameters
    ----------
    rows : Integer
        Pixel size of the height of a "noise image"
    columns : Integer
        Pixel size of the width of an "noise image"
    random_noise_dimension : Integer
        Number of channels of a "noise image"

    Returns
    -------
    2-dim numpy array
        Array of shape (rows * columns, random_noise_dimensions) with
        normally distributed values with mean 0 and standard deviation 1

    """
    return np.random.normal(0, 1, (rows * columns, random_noise_dimension))