"""
Author: Pascal Pilz
Matr.Nr.: K12111234
Exercise 3
"""
import glob
import os
from PIL import Image
import numpy as np


class ImageStandardizer:
    def __init__(self, input_dir):
        # We convert the input path into an absolute path, in case we got a relative path
        self.input_dir = os.path.abspath(input_dir)
        self.files = glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True)
        # If we didn't find anything we raise a ValueError
        if not self.files:
            raise ValueError(f"No .jpg files found in {os.path.relpath(input_dir)}")
        self.files.sort()
        self.mean = None
        self.std = None

    def analyze_images(self):
        total_means = np.array([0.0, 0.0, 0.0])
        total_stds = np.array([0.0, 0.0, 0.0])
        for file in self.files:
            with Image.open(file, "r") as image:
                im = np.asarray(image)
            # We are saving the mean so that we can use it to calculate the variance
            im_mean = im.sum(axis=(0, 1)) / (im.shape[0] * im.shape[1])
            total_means += im_mean
            # We use this formula for the variance instead of the simplified one because it seems like with the
            # simplified one the because otherwise the numbers get too large with the squared sum and we experience
            # integer overflow. Apparently this would not happen in pure python, but it does happen in numpy
            total_stds += np.sqrt(np.square(im - im_mean).sum(axis=(0, 1)) / (im.shape[0] * im.shape[1]))
        self.mean = total_means / len(self.files)
        self.std = total_stds / len(self.files)
        return self.mean, self.std

    def get_standardized_images(self):
        if self.mean is None or self.std is None:
            raise ValueError(f"self.mean or self.std is None")
        for file in self.files:
            with Image.open(file, "r") as image:
                im = np.asarray(image).astype("float64")
            im = (im - self.mean.astype("float64")) / self.std.astype("float64")
            yield im.astype("float32")
