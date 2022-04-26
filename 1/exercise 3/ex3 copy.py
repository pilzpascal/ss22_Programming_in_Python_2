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
        self.input_dir = os.path.abspath(input_dir)
        self.files = glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True)
        if not self.files:
            raise ValueError(f"No .jpg files found in {os.path.relpath(input_dir)}")
        self.files.sort()
        self.mean = None
        self.std = None

    def analyze_images(self):
        total_sum = np.array([0.0, 0.0, 0.0])
        total_sum_sq = np.array([0.0, 0.0, 0.0])
        # Count represents the total pixel count
        count = 0
        for file in self.files:
            with Image.open(file, "r") as image:
                im = np.asarray(image)
            total_sum += im.sum(axis=(0, 1))
            total_sum_sq += (np.square(im)).sum(axis=(0, 1))
            # We sum up the pixels of each image
            count += im.shape[0] * im.shape[1]
            print(os.path.basename(file), im.shape[0], im.shape[1], im.shape[0] * im.shape[1], count, total_sum)
        self.mean = total_sum / count
        self.std = np.sqrt((total_sum_sq / count) - (np.square(self.mean)))
        return self.mean, self.std
