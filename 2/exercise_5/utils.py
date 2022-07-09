# -*- coding: utf-8 -*-
"""exercise_5/utils.py

Author -- Pascal Pilz (original authors: Michael Widrich, Andreas Schörgenhumer)
Contact -- pasc.pilz@gmail.com
Date --

###############################################################################

This is the utils file of the project. The rough outline is taken form the example project
provided to us, the original authors are Michael Widrich and Andreas Schörgenhumer.

###############################################################################

Utils file of project.
"""

import os

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from matplotlib import pyplot as plt

from math import ceil
from PIL import Image
from typing import Union
import shutil


def evaluate_model(model: torch.nn.Module, data_loader: torch.utils.data.dataloader, loss_fn, device: torch.device,):
    """Function for evaluation of a model `model` on the data in `data_loader` on device `device`,
    using the specified `loss_fn` loss function"""
    model.eval()
    # We will accumulate the mean loss in variable `loss`
    loss = 0
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `data_loader`
        for data in tqdm(data_loader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs, targets, file_names = data
            inputs = torch.stack([torch.tensor(elem) if not torch.is_tensor(elem) else elem for elem in inputs])
            inputs = inputs.to(device)

            # Get outputs of the specified model
            outputs = model(inputs)

            # Add the current loss, which is the mean loss over all minibatch samples
            # (unless explicitly otherwise specified when creating the loss function!)
            loss += loss_fn(outputs, targets).item()
    # Get final mean loss by dividing by the number of minibatch iterations (which
    # we summed up in the above loop)
    loss /= len(data_loader)
    model.train()
    return loss


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            ax.imshow(data[i, 0], cmap="gray", interpolation="none")
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=100)

    plt.close(fig)


def validate_images(input_dir: str, output_dir: str, log_file: str, formatter: str = "06d") -> int:
    max_file_size = 250000
    min_pixel_size = 96
    invalid_files = []
    hashes = []
    file_names = []
    valid_images = 0

    formatter = ''.join(["{:", formatter, "}"])
    # Converting the formatter into the correct form; for example '06d' becomes '{:06d}'

    os.makedirs(os.path.relpath(output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    input_dir = os.path.abspath(input_dir)
    # Here we use os.walk instead of glob.glob because with glob.glob i did not find a good way of getting all file
    # types while not getting directories, and this is made easier with os.walk
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Storing tuples of the path and the filename in order to keep them separated, which we need for sorting
            file_names.append(os.path.join(root, file))
    file_names.sort()

    for f in file_names:
        # To avoid checking .DS_Store files we implement this condition
        # if os.path.relpath(f, start=input_dir).startswith("."):
        # continue

        # Checking Rule 1: files must end with .jpg, .JPG, .jpeg, .JPEG
        if not (f.endswith(".jpg") or f.endswith(".JPG") or f.endswith(".jpeg") or f.endswith(".JPEG")):
            error_code = 1
            invalid_files.append(f"{os.path.relpath(f, start=input_dir)};{error_code}\n")
            continue

        # Checking Rule 2: files must be <= 250000 bits
        if os.path.getsize(f) > max_file_size:
            error_code = 2
            invalid_files.append(f"{os.path.relpath(f, start=input_dir)};{error_code}\n")
            continue

        # Opening the image in a try-except block, such that i can catch the exception if Image.open fails and thus
        # Checking Rule 3: files must be readable as images
        try:
            image = Image.open(f, mode="r")
            # Checking Rule 4: images must be of size (H,W,3), with H,W >= 96
            if image.mode != "RGB" or image.size[0] < min_pixel_size or image.size[1] < min_pixel_size:
                error_code = 4
                invalid_files.append(f"{os.path.relpath(f, start=input_dir)};{error_code}\n")
                image.close()
                continue
            image_as_array = np.array(image)

            # Checking Rule 5: image variance > 0, i.e., images cannot be just one color
            # I am doing this by converting them into grayscale and just taking the variance over the result
            if np.var(image.convert('L')) == 0:
                error_code = 5
                invalid_files.append(f"{os.path.relpath(f, start=input_dir)};{error_code}\n")
                image.close()
                continue

            image.close()
        except Exception:
            # We have to catch "Except", as the exception that is raised if Image.open fails does not inherit from
            # BaseException
            error_code = 3
            invalid_files.append(f"{os.path.relpath(f, start=input_dir)};{error_code}\n")
            continue
        else:
            image.close()

        # Hashing the images; converting them to string makes it immutable and thus hashable
        im_hash = hash(image_as_array.tostring())
        # Checking Rule 6: no images shall be copied more than once
        if im_hash in hashes:
            error_code = 6
            invalid_files.append(f"{os.path.relpath(f, start=input_dir)};{error_code}\n")
            continue
        hashes.append(im_hash)

        # Preparing the file name/path
        file_name = os.path.relpath(os.path.join(output_dir, formatter.format(valid_images) + ".jpg"))
        # We copy the images instead of saving them manually in order to avoid compression artefacts and the like
        shutil.copy(f, file_name)
        # This counter is used for the return statement, but also for numbering the checked and copied files
        valid_images += 1

    # Writing the log file
    with open(log_file, "a") as f:
        for line in invalid_files:
            f.write(line)
    return valid_images


class ImageStandardizer:
    def __init__(self, files: list):
        # as input we get a list of file names
        self.files = files
        self.mean = None
        self.std = None

    def analyze_images(self):
        # Initializing two array with a length corresponding to the number of files; and a width of 3, corresponding
        # to the three color channels
        means = np.zeros(shape=(len(self.files), 3), dtype=np.float64)
        stds = np.zeros(shape=(len(self.files), 3), dtype=np.float64)
        for index, file in enumerate(self.files):
            with Image.open(file, "r") as image:
                im = np.asarray(image)
            # Using numpy's built in methods for mean and standard deviation, along axis (0, 1), as to preserve
            # the individual RGB entries
            means[index] = im.mean(axis=(0, 1))
            stds[index] = im.std(axis=(0, 1))
        # Averaging over the means and the standard deviations
        self.mean = means.mean(axis=0)
        self.std = stds.mean(axis=0)
        return self.mean, self.std

    def get_standardized_images(self):
        if self.mean is None or self.std is None:
            raise ValueError(f"self.mean or self.std is None")
        for file in self.files:
            with Image.open(file, "r") as image:
                im = np.asarray(image).astype("float64")
            im = (im - self.mean.astype("float64")) / self.std.astype("float64")
            yield im.astype("float32")


def get_obscured_image(image_array: np.ndarray, offset: (int, int), spacing: (int, int),
                       mean: Union[list, torch.Tensor], std: Union[list, torch.Tensor]):
    """
        A modified version of the function ex4 we wrote for exercise 4 of assignment 2.
        This function returns the original image and the obscured image.

                Parameters:
                        image_array (ndarray): A three dimensional numpy array of shape (M, N, 3)
                        offset (2-tuple of int): The values by which to offset the grid
                        spacing (2-tuple of int): The spacing of the grid
                        mean (float): the arithmetic mean over the training set, used to normalise the images
                        std (float): the standard deviation over the training set, used to normalise the images
                Returns:
                        2-tuple of ndarray

                        input_array (ndarray): A numpy array of shape (3, M, N) where all pixels outside the grid are
                        set to 0

                        known_array (ndarray): A numpy array of shape (3, M, N) where all pixels which are not 0 in
                        the known_array are equal to the corresponding pixels of the image_array, i.e., an inverted
                        boolean mask of the inverted known_array applied to image_array
    """

    offset = tuple(int(entry) for entry in offset)
    spacing = tuple(int(entry) for entry in spacing)

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
    # remaining_pixels =
    # ceil((length_rows - offset_rows) / spacing_rows) * ceil((length_cols - offset_cols) / spacing_cols)
    if not ceil((image_array.shape[1] - offset[0]) / spacing[0]) *\
            ceil((image_array.shape[0] - offset[1]) / spacing[1]) >= 144:
        raise ValueError(f"The number of known pixels after removing must be at least 144 but is \
    {ceil((image_array.shape[1] - offset[0]) / spacing[0]) * ceil((image_array.shape[0] - offset[1]) / spacing[1])}")

    # Before the transpose we have input_array.shape = image_array.shape = (M, N, 3)
    # Creating a working copy of the image, which is also transposed immediately
    input_array = np.transpose(image_array, (2, 0, 1)).copy()
    # After the transpose we have input_array.shape = (3, M, N)

    # normalizing the input array separately
    input_array = np.array([(elem - mean[i]) / std[i] for i, elem in enumerate(input_array)])

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

    # Basically we iterate over the pixels, considering the RGB values for each and if one of the RGB is greater than 0
    # we set the entire pixel to 1, so [0, 121, 54] would now become [1, 1, 1]
    # known_array.shape = (3, M, N)
    for i in range(known_array.shape[1]):
        for j in range(known_array.shape[2]):
            if np.any(known_array[:, i, j]):
                known_array[:, i, j] = 1

    # input_array.shape = (3, M, N), known_array.shape = (3, M, N)
    # where M, N are from  image_array (original image) with image_array.shape = (M, N, 3)
    return input_array, known_array


def get_target_array(image_array: Union[np.ndarray, torch.Tensor], known_array: Union[np.ndarray, torch.Tensor]):
    """
        A modified version of the function ex4 we wrote for exercise 4 of assignment 2.
        This function returns the target array.

                Parameters:
                        image_array (ndarray): A three dimensional numpy array of shape (M, N, 3)
                        known_array (ndarray): A numpy array of shape (3, M, N) where all pixels which are not 0 in
                        the known_array are equal to the corresponding pixels of the image_array, i.e., an inverted
                        boolean mask of the inverted known_array applied to image_array
                Returns:
                        target_array (ndarray): A numpy array of shape (remaining_pixels * 3,), where remaining_pixels
                        are the non 0 pixels in input_array. target_array has the pixel values of image_array
                        (the original image) everywhere the image_array got overwritten with 0, it is flattened out,
                        i.e., one dimensional and of length (remaining_pixels * 3). First a sequence of all R values,
                        then all G values, then all B values.
    """
    # target_array has the pixel values of image_array (the original image) everywhere the image_array got
    # overwritten with 0, it is flattened out, i.e., one dimensional and of length (remaining_pixels * 3). First a
    # sequence of all R values, then all G values, then all B values.
    target_array = np.transpose(image_array, (2, 0, 1))[known_array < 1].copy()
    return target_array


def set_seed(seed: Union[float, int] = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
