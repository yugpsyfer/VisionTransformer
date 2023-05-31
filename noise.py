"""
Was made to add gaussian noise in a clean daatset then I ended up using the raindrop dataset.
"""


import torch


class Noise():
    def __init__(self, noise_type) -> None:
        self.noise_type= noise_type


    def add_noise(self, input):
        if self.noise_type == 'gauss':
            return self.__add_gaussian_noise__(input)

    def __add_gaussian_noise__(self, x):
        mean = 0
        std = 1.0

        return x + torch.randn(x.size()) * std + mean

