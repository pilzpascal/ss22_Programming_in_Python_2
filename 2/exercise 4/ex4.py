"""
Author: Pascal Pilz
Matr.Nr.: K12111234
Exercise 4
"""

import numpy as np


def ex4(image_array, offset, spacing):
    # Casting offset and spacing to int, if this fails we get a ValueError, which is what the assignments specifies
    offset = tuple(int(entry) for entry in offset)
    spacing = tuple(int(entry) for entry in spacing)
    # Checking the other rules that the assignment specifies, the final one
    # about the number of remaining pixels will be checked later
    if not isinstance(image_array, np.ndarray):
        raise TypeError(f"image_array is of type {type(image_array)}, not np.ndarray.")
    if not (image_array.shape[2] == 3 or image_array.ndim == 3):
        raise NotImplementedError("image_array is not a 3D array.")
    for i in offset:
        if i < 0 or i > 32:
            raise ValueError("offset is not in [0,32].")
    for i in spacing:
        if i < 2 or i > 8:
            raise ValueError("Spacing is not in [2,8].")

    # Creating a working copy of the image, which also transposed immediately
    input_array = np.transpose(image_array, (2, 0, 1)).copy()

    # Setting the offset rows and cols to 0
    # N
    input_array[:, :, :offset[0]] = 0
    # M
    input_array[:, :offset[1], :] = 0

    # N
    for i in range(offset[0] + 1, input_array.shape[2]):
        # With this condition we make every row that is not a multiple of the row spacing to 0
        if not i % spacing[0] == 1:
            input_array[:, :, i] = 0

    # M
    for i in range(offset[1] + 1, input_array.shape[1]):
        # With this condition we make every col that is not a multiple of the col spacing to 0
        if not i % spacing[1] == 1:
            input_array[:, i, :] = 0

    # We get the array of known pixels, represented by 0 if unknown, 1 if known.
    # This should be of same shape as image_array, therefore we transpose again.
    known_array = np.transpose(input_array, (1, 2, 0)).copy()
    known_array[known_array > 0] = 1

    # We check if the number of remaining pixels is grater than 144
    if np.sum(known_array) < 144:
        raise ValueError("Number of remaining pixels is less than 144")

    # target_array is equal to the image array
    target_array = image_array[not known_array].flatten()

    return input_array, known_array, target_array
