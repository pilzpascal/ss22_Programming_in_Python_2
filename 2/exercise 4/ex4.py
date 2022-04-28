"""
Author: Pascal Pilz
Matr.Nr.: K12111234
Exercise 4
"""

import numpy as np
# We only need the ceil function to round up for calculating the remaining pixels when checking the specifications
from math import ceil


def ex4(image_array, offset, spacing):
    """
        Modifies an input image, using the given offset and spacing, in order to create training data for our
        image-upscaling project. For more detail, see as2.pdf

                Parameters:
                        image_array (ndarray): A three dimensional numpy array of shape (M, N, 3)
                        offset (2-tuple of int): The values by which to offset the grid
                        spacing (2-tuple of int): The spacing of the grid
                Returns:
                        3-tuple of ndarray

                        input_array (ndarray): A numpy array of shape (3, M, N) where all pixels outside the grid are
                        set to 0

                        known_array (ndarray): A numpy array of shape (3, M, N) where all pixels which are not 0 in
                        the known_array are equal to the corresponding pixels of the image_array, i.e., an inverted
                        boolean mask of the inverted known_array applied to image_array

                        target_array (ndarray): A numpy array of shape (remaining_pixels * 3,), where remaining_pixels
                        are the non 0 pixels in input_array. target_array has the pixel values of image_array
                        (the original image) everywhere the image_array got overwritten with 0, it is flattened out,
                        i.e., one dimensional and of length (remaining_pixels * 3). First a sequence of all R values,
                        then all G values, then all B values.
    """

    # Casting offset and spacing to int, if this fails we get a ValueError, which is what the assignments specifies
    offset = tuple(int(entry) for entry in offset)
    spacing = tuple(int(entry) for entry in spacing)
    # Checking the other rules that the assignment specifies
    if not isinstance(image_array, np.ndarray):
        raise TypeError(f"image_array is of type {type(image_array)}, not np.ndarray.")
    if not (image_array.ndim == 3 and image_array.shape[2] == 3):
        raise NotImplementedError("image_array is not a 3D array.")
    for i in offset:
        if not 0 <= i <= 32:
            raise ValueError("offset is not in [0,32].")
    for i in spacing:
        if not 2 <= i <= 8:
            raise ValueError("Spacing is not in [2,8].")
    # We check if the number of remaining pixels is grater than 144. The formula I use for this is:
# remaining_pixels = ceil((length_rows - offset_rows) / spacing_rows) * ceil((length_cols - offset_cols) / spacing_cols)
    if not ceil((image_array.shape[1] - offset[0]) / spacing[0]) *\
            ceil((image_array.shape[0] - offset[1]) / spacing[1]) >= 144:
        raise ValueError(f"The number of known pixels after removing must be at least 144 but is \
{ceil((image_array.shape[1] - offset[0]) / spacing[0]) * ceil((image_array.shape[0] - offset[1]) / spacing[1])}")

    # Before the transpose we have input_array.shape = image_array.shape = (M, N, 3)
    # Creating a working copy of the image, which is also transposed immediately
    input_array = np.transpose(image_array, (2, 0, 1)).copy()
    # After the transpose we have input_array.shape = (3, M, N)

    # Setting the offset rows and cols to 0
    # N
    input_array[:, :, :offset[0]] = 0
    # M
    input_array[:, :offset[1], :] = 0

    # Setting the rows/cols outside of the specified grid to 0
    # N
    for i in range(input_array.shape[2] - offset[0]):
        # With this condition we make every row that is not a multiple of the row spacing to 0
        if not i % spacing[0] == 0:
            input_array[:, :, (i + offset[0])] = 0
    # M
    for i in range(input_array.shape[1] - offset[1]):
        # With this condition we make every col that is not a multiple of the col spacing to 0
        if not i % spacing[1] == 0:
            input_array[:, (i + offset[1]), :] = 0

    # We get the array of known pixels, represented by 0 if unknown, 1 if known.
    # This should be of same shape as image_array, therefore we transpose again.
    known_array = input_array.copy()

    # This was my previous attempt, but we are missing something: if we have a pixel like [0, 121, 54] then the pixel
    # would become [0, 1, 1] but it should be [1, 1, 1].
    # known_array[known_array > 0] = 1

    # This is the new version; it works, but it is not very elegant.
    # Basically we iterate over the pixels, considering the RGB values for each and if one of the RGB is greater than 0
    # we set the entire pixel to 1, so [0, 121, 54] would now become [1, 1, 1]
    # known_array.shape = (3, M, N)
    for i in range(known_array.shape[1]):
        for j in range(known_array.shape[2]):
            if np.any(known_array[:, i, j]):
                known_array[:, i, j] = 1

    # target_array has the pixel values of image_array (the original image) everywhere the image_array got
    # overwritten with 0, it is flattened out, i.e., one dimensional and of length (remaining_pixels * 3). First a
    # sequence of all R values, then all G values, then all B values.
    target_array = np.transpose(image_array, (2, 0, 1))[known_array < 1].copy()

    # input_array.shape = (3, M, N), known_array.shape = (3, M, N), target_array.shape = (remaining_pixels * 3,)
    # where M, N are from  image_array (original image) with image_array.shape = (M, N, 3)
    return input_array, known_array, target_array
