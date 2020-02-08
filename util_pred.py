from tile_helpers import *
from segment import segment_f


def predict_test(im, net):
    """
    Takes entire image, breakes it into tiles, calculates
    predictions, puts it back together,
    returns predictions for all pixels of the orinal image.
    """
    tile_size = 512
    if np.min([im.cpu().detach().numpy().shape[2], im.cpu().detach().numpy().shape[3]]) < 512:
        tile_size = np.min(im.cpu().detach().numpy().shape[2], im.cpu().detach().numpy().shape[3])

    tile_list = tile_image(im.cpu().detach().numpy()[0][0], tile_size)
    tol = []
    for t in tile_list:
        with torch.no_grad():
            pre = net(t)
        tol.append(pre)
    out = untile_image(tol, im.cpu().detach().numpy()[0][0].shape)
    out = np.expand_dims(np.expand_dims(out, 0), 0)
    return torch.from_numpy(out).cuda()


def predict_train(im, mask, tile_size):
    if np.min([im.cpu().detach().numpy().shape[2], im.cpu().detach().numpy().shape[3]]) < tile_size:
        tile_size = np.min([im.shape[2], im.shape[3]])
    tile_img = tile_image_train(im[0][0], tile_size)
    tile_mask = tile_image_train(mask[0][0], tile_size)
    return tile_img, tile_mask


def threshold(pred, param):
    """
    Takes the predicted image, thresholds it with the determined
    param, returns binary image.
    """
    return pred > param


def segment(th):
    """
    Takes thresholded image and segments it into the individual
    cells. Returns mask where every
    individual cell is assigned a unique label.
    """
    return segment_f(th)
