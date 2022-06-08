# -*- coding: utf-8 -*-
"""exercise_5/main.py

Author -- Pascal Pilz (original authors: Michael Widrich, Andreas Schörgenhumer)
Contact -- pasc.pilz@gmail.com
Date --

###############################################################################

This is the main file of the project. The rough outline is taken form the example project
provided to us, the original authors are Michael Widrich and Andreas Schörgenhumer.

###############################################################################

Main file of project.
"""

import os

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import evaluate_model
from utils import plot

from datasets import RotatedImages


def main(results_path, network_config: dict, lr: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = 50_000, device: torch.device = torch.device("cuda:0")):
    """Main function that takes hyperparameters and performs training and evaluation of model"""
    # Set a known random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Prepare a path to plot to
    plot_path = os.path.join(results_path, "plots")
    os.makedirs(plot_path, exist_ok=True)

    # TODO: Load dataset

    # TODO: Split dataset into training, validation and test set

    # TODO: use random rotation and further data augmentation
    # Create datasets and data_loaders with rotated targets without augmentation (for evaluation)
    training_set_eval = RotatedImages(dataset=training_set, rotation_angle=45)
    validation_set = RotatedImages(dataset=validation_set, rotation_angle=45)
    test_set = RotatedImages(dataset=test_set, rotation_angle=45)
    train_loader = torch.utils.data.DataLoader(training_set_eval, batch_size=1, shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    # # TODO: use random rotation and further data augmentation
    # Create datasets and data_loaders with rotated targets with augmentation (for training)
    training_set_augmented = RotatedImages(dataset=training_set, rotation_angle=45,
                                           transform_chain=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                               transforms.RandomVerticalFlip()]))
    train_loader_augmented = torch.utils.data.DataLoader(training_set_augmented, batch_size=16, shuffle=True,
                                                         num_workers=0)

    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))

    # TODO: do the thing


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config_file) as cf:
        config = json.load(cf)
    main(**config)
