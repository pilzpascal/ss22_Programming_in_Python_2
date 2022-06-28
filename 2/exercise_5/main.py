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

    # # TODO: use random rotation and further data augmentation create a better data augmentation chain
    # Create datasets and data_loaders with rotated targets with augmentation (for training)
    training_set_augmented = RotatedImages(dataset=training_set, rotation_angle=45,
                                           transform_chain=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                               transforms.RandomVerticalFlip()]))
    train_loader_augmented = torch.utils.data.DataLoader(training_set_augmented, batch_size=16, shuffle=True,
                                                         num_workers=0)

    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))

    # Create Network
    net = Net1(**network_config)
    net.to(device)

    # Get MSE loss function
    mse = torch.nn.MSELoss()

    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # parameters for training
    print_stats_at = 100  # print status to tensorboard every x updates
    plot_at = 10_000  # plot every x updates
    validate_at = 5_000  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    saved_model_file = os.path.join(results_path, "best_model.pt")
    torch.save(net, saved_model_file)

    # TODO: training


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config_file) as cf:
        config = json.load(cf)
    main(**config)
