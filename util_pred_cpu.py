import numpy as np
import torch
import torch.nn as nn
from tile_helpers_cpu import *
from segment import segment_f


def threshold(pred, param):
    """
    Takes the predicted image, thresholds it with the determined
    param, returns binary image.
    """
    return pred[pred>param]


def segment(th):
    """
    Takes thresholded image and segments it into the individual
    cells. Returns mask where every
    individual cell is assigned a unique label.
    """
    return segment_f(th)
