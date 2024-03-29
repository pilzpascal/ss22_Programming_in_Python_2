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
