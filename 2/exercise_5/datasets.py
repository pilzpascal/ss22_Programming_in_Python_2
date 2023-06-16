# -*- coding: utf-8 -*-
"""exercise_5/datasets.py

Author -- Pascal Pilz (original authors: Michael Widrich, Andreas Schörgenhumer)
Contact -- pasc.pilz@gmail.com
Date --

###############################################################################

This is the file containing the datasets and useful functions. The rough outline is taken form the example project
provided to us, the original authors are Michael Widrich and Andreas Schörgenhumer.

###############################################################################

Datasets file of project.
"""

import torch
import torchvision.transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image
from typing import Union

import utils


_MAX_OFFSET = 8

_MIN_SPACING = 2
_MAX_SPACING = 6


class ImageData(Dataset):
    def __init__(self, file_names: list, transform_chain: torchvision.transforms.Compose = None, seed: int = 0):
        utils.set_seed(seed)
        self.file_names = file_names
        self.transform_chain = transform_chain

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        return self.file_names[index], index


class TransformedImages(Dataset):
    def __init__(self,
                 data_set: torch.utils.data.Dataset,
                 transform_chain: torchvision.transforms.Compose = None,
                 mean: Union[list, torch.Tensor] = torch.tensor([0, 0, 0]),
                 std: Union[list, torch.Tensor] = torch.tensor([1, 1, 1]),
                 seed: int = 0,
                 random: bool = True):
        self.data_set = data_set
        self.transform_chain = transform_chain
        self.mean = mean
        self.std = std
        self.seed = seed
        self.random = random

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        path, i = self.data_set[index]
        with Image.open(path, "r") as image:
            if self.transform_chain is not None:
                image_data = self.transform_chain(image)
            else:
                image_data = TF.center_crop(img=image, output_size=[100, 100])

        if self.random:
            # get random offset and spacing, apply to image using the function from ex4 to get the training data
            offset = (np.random.randint(_MAX_OFFSET + 1),
                      np.random.randint(_MAX_OFFSET + 1))
            spacing = (np.random.randint(_MIN_SPACING, _MAX_SPACING + 1),
                       np.random.randint(_MIN_SPACING, _MAX_SPACING + 1))
        else:
            utils.set_seed(self.seed)
            offset = (_MAX_OFFSET//2, _MAX_OFFSET//2)
            spacing = (_MAX_SPACING//2, _MAX_SPACING//2)

        # transform to numpy array and normalise according to training data mean and standard deviation
        image_data = np.asarray(image_data, dtype=np.float32)

        # input_array is the image with lines blacked out corresponding to offset and spacing, see ex4 for more info
        # know_array is a list of the pixels that are not blacked out
        input_array, known_array = utils.get_obscured_image(image_data, offset, spacing, self.mean, self.std)

        image_data = torch.from_numpy(image_data.transpose((2, 0, 1)))
        input_array = torch.from_numpy(input_array)
        known_array = torch.from_numpy(np.array([known_array[0]]))

        full_inputs = torch.concat([input_array, known_array])

        # full_inputs contains the the full inputs to the model,
        # including the image itself and information about the known pixels.
        # image_data is the original image transposed from (M, N, 3) to (3, M, N)
        return full_inputs, image_data, i


# custom collate, specifically for padding the target arrays with trailing 0s
def torch_tensor_collate(data):
    sequences = [elem[1] for elem in data]
    max_len = np.max([seq.shape[0] for seq in sequences])
    output_sequences = torch.full(size=(len(sequences), max_len), fill_value=-1, dtype=torch.float32)
    for index, elem in enumerate(output_sequences):
        elem[:sequences[index].shape[0]] = sequences[index]
    return [(elem[0], output_sequences[i], elem[2]) for i, elem in enumerate(data)]


def simple_list_collate(data):
    return [elem[0] for elem in data], [elem[1] for elem in data], [elem[2] for elem in data]
