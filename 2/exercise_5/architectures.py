# -*- coding: utf-8 -*-
"""exercise_5/architectures.py

Author -- Pascal Pilz (original authors: Michael Widrich, Andreas Schörgenhumer)
Contact -- pasc.pilz@gmail.com
Date --

###############################################################################

This is the file containing the different architectures. The rough outline is taken form the example project
provided to us, the original authors are Michael Widrich and Andreas Schörgenhumer.

###############################################################################

Architectures file of project.
"""

import torch


class SimpleCNN(torch.nn.Module):
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super().__init__()

        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=(kernel_size, kernel_size),
                padding=int(kernel_size / 2)
            ))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)

        self.output_layer = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=1,
            kernel_size=(kernel_size, kernel_size),
            padding=int(kernel_size / 2)
        )

    def forward(self, x):
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        return pred


class CNN1(torch.nn.Module):
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        super().__init__()

        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=(kernel_size, kernel_size),
                padding=int(kernel_size / 2)
            ))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)

        self.output_layer = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=3,
            kernel_size=(kernel_size, kernel_size),
            padding=int(kernel_size / 2)
        )

    def forward(self, x):
        cnn_out = self.hidden_layers(x)
        predictions = self.output_layer(cnn_out)
        return predictions
