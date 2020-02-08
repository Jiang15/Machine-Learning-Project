from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from matplotlib import pyplot as plt
import numpy as np
import torch
from skimage.filters import gaussian


def segment_f(th, min_distance=6, topology=lambda x: gaussian(-x, sigma=0.7)):
    """
    Performs watershed segmentation on thresholded image. Seeds have to
    have minimal distance of min_distance. topology defines the watershed
    topology to be used, default is the negative distance transform. Can
    either be an array with the same size af th, or a function that will
    be applied to the distance transform.
    """
    dtr = ndi.morphology.distance_transform_edt(th.cpu().detach().numpy()[0])
    if topology is None:
        topology = -dtr
    elif callable(topology):
        topology = topology(dtr)
    m = peak_local_max(-topology, min_distance, indices=False)
    m_lab = ndi.label(m)[0]
    wsh = watershed(topology, m_lab, mask=th.cpu().detach().numpy()[0])
    wsh = np.expand_dims(wsh, axis=0)
    return torch.from_numpy(np.expand_dims(wsh, axis=0))
