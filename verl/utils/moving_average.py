# alpha-dpg
# Copyright (C) 2026 Naver Corporation. All rights reserved.

import numpy as np
import torch

def average(moving_averages):
    if moving_averages.items():
        weighted_values, weights = list(zip(
                *[(ma.value * ma.weight, ma.weight) for _, ma in moving_averages.items()]
            ))
        return sum(weighted_values) / sum(weights)
    else:
        return 0

class ExponentiallyWeightedMovingAverage:
    def __init__(self, ewma_weight=0.99):
        self.value = None
        self.ewma_weight = ewma_weight

    def update(self, new):
        if isinstance(new, np.ndarray):
            new = np.mean(new)
        elif isinstance(new, torch.Tensor):
            new = new.mean()
        else:
            new = new
        if self.value:
            self.value = self.ewma_weight * self.value + \
                    (1 - self.ewma_weight) * new
        else:
            self.value = new

class WindowedMovingAverage:
    """
    Keeps a moving average of a quantity over a fixed-size window.
    """
    def __init__(self, window_size=1000):
        """
        Parameters
        ----------
        window_size: int
            The number of samples to average over.
        """
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        self.window_size = window_size
        self.buffer = []
        self.value = None  # Stores the average of the last completed window.

    def update(self, new):
        """
        Adds new values to the buffer and updates the average

        Parameters
        ----------
        new: torch.Tensor, np.ndarray, list, or float
            A collection of new pointwise estimates to add to the buffer.
        """
        # Ensure 'new' is a flat list of numbers
        if isinstance(new, torch.Tensor):
            new_values = new.flatten().tolist()
        elif isinstance(new, np.ndarray):
            new_values = new.flatten().tolist()
        elif hasattr(new, '__iter__'):
            new_values = list(new)
        else:
            new_values = [new]  # Treat a single number as a list

        self.buffer.extend(new_values)

        # crop to window size
        self.buffer = self.buffer[-self.window_size:]

        # Calculate the mean of the full window and update the public value
        self.value = sum(self.buffer) / len(self.buffer)

class MovingAverage(object):
    """
    Keeps a moving average of some quantity
    """
    def __init__(self):
        """
        Parameters
        ----------
        init_value: float
            the initial value of the moving average
        """
        self.value = None
        self.weight = 0

    def __iadd__(self, new_values):
        """
        Includes an array of values into the moving average

        Parameters
        ----------
        new_values: torch/np array of floats
            pointwise estimates of the quantity to estimate
        """
        self.update_array(new_values)
        return self

    def update_array(self, new_values):
        """
        Includes an array of values into the moving average

        Parameters
        ----------
        new_values: torch/np array of floats
            pointwise estimates of the quantity to estimate
        """
        new_value = new_values.mean()
        new_weight = len(new_values)
        self.add(new_value, new_weight)

    def update(self, new_value, new_weight=1):
        """
        Includes a single value into a moving average

        Parameters
        ----------
        new_value: float
            pointwise estimate of the quantity to estimate
        new_weight: float
            a weight by which to ponder the new value
        """
        if self.value is None:
            self.value = new_value
        else:
            self.value = (self.value * self.weight + new_weight * new_value) / \
                    (self.weight + new_weight)
        self.weight += new_weight

    def reset(self):
        """
        Resets the moving average to its initial conditions
        """
        self.value = None
        self.weight = 0
