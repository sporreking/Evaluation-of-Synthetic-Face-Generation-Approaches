from src.generator.Generator import Generator
import torch
from torch import nn
import math


class Flatten(nn.Module):
    # nn.Flatten implementation (for backward compatibility)
    def forward(self, input):
        return input.view(input.size(0), -1)


def get_dec_arch(gen: Generator) -> nn.Sequential:
    """
    Get decoder architecture associated with given generator.

    Args:
        gen (Generator): Generator associated with the decoder.

    Returns:
        nn.Sequential: Decoder architecture.
    """
    # As defined in the paper.
    len_z = len(gen.latent_space_mean())
    h_size = math.floor(len_z / 2)
    decoder = nn.Sequential(
        nn.Linear(len_z, h_size),
        nn.ReLU(inplace=True),
        nn.Linear(h_size, len_z),
    )
    return decoder


def get_cls_arch() -> nn.Sequential:
    """
    Get classifier architecture.

    Returns:
        nn.Sequential: Classifier architecture.
    """
    # As defined in the paper.

    classifier = nn.Sequential(
        # in: 3 x 256 x 256
        nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False),
        nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        # out: 64 x 128 x 128
        nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
        nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        # out: 64 x 64 x 64
        nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
        nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        # out: 64 x 64 x 64
        nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
        nn.AdaptiveAvgPool2d(8),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        # out: 64 x 8 x 8
        # FC
        Flatten(),
        nn.Linear(4096, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(256, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    return classifier
