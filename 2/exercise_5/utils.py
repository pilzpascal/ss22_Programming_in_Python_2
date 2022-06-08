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

import torch
import torch.utils.data
from tqdm import tqdm
from matplotlib import pyplot as plt


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


# TODO: make collate function for NxMx3 sized data
def custom_collate(data, ):
    max_len = max([len(s[1].data[1]) for s in data])
