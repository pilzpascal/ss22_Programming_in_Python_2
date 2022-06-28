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

from PIL import Image
import shutil
import glob


def evaluate_model(model: torch.nn.Module, data_loader: torch.utils.data.dataloader, loss_fn, device: torch.device):
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
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get outputs of the specified model
            outputs = model(inputs)

            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance

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
    def __init__(self, files):
        # as input we get a list of file names
        self.files = files
        # If we didn't find anything we raise a ValueError
        if not self.files:
            raise ValueError(f"No .jpg files found in {os.path.relpath(files[0])}")
        self.files.sort()
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



# TODO: make collate function for NxMx3 sized data
# I am no longer sure if we actually need this, since the data is always of shape 100x100
def custom_collate(data, ):
    sequences = [sample[0] for sample in data]
    max_len = np.max([seq.shape[0] for seq in data])
