import math
import numpy as np
import torch


def get_tile_pos(imshape, tilesize):
    """
    Given a image shape and a tilesize, returns a list of tuples with the (x,y) offset
    for the nth tile. Assumes square tiles.
    """

    offset = tilesize//2

    (rowpx, colpx) = imshape
    nrow = math.ceil((rowpx-tilesize) / offset) +1
    ncol = math.ceil((rowpx-tilesize) / offset) +1

    out = []
    for row in range(nrow):
        for col in range(ncol):
            if row < nrow-1:
                x = row*offset
            else:
                x = rowpx - tilesize
            if col < ncol-1:
                y = col*offset
            else:
                y = colpx - tilesize
            out.append((x,y))

    return out

def get_tile_pos_train(imshape, tilesize):
    """
    Given a image shape and a tilesize, returns a list of tuples with the (x,y) offset
    for the nth tile. Assumes square tiles.
    """

    offset = tilesize//2

    (rowpx, colpx) = imshape
    nrow = math.ceil((rowpx-tilesize) / offset) + 1
    ncol = math.ceil((rowpx-tilesize) / offset) + 1

    out = []
    for row in range(nrow):
        for col in range(ncol):
            if row < nrow-1:
                x = row*offset
            else:
                x = rowpx - tilesize
            if col < ncol-1:
                y = col*offset
            else:
                y = colpx - tilesize
            out.append((x,y))

    return out


def tile_image(im, tilesize):
    """
    Break image into tiles of given tilesize. Tiles are chosen to have large
    overlap of tilesize/2 with previous and next tile for good sticking together
    """
    tile_pos = get_tile_pos(im.shape, tilesize)
    def get_tile(tup):
        (r, c) = tup
        return torch.from_numpy(np.expand_dims(np.expand_dims(im[r:r+tilesize, c:c+tilesize],0),0)).cuda()

    return [get_tile(tup) for tup in tile_pos]


def tile_image_train(im, tilesize):
    """
    Break image into tiles of given tilesize. Tiles are chosen to have large
    overlap of tilesize/2 with previous and next tile for good sticking together
    """
    tile_pos = get_tile_pos_train(im.shape, tilesize)
    def get_tile(tup):
        (r, c) = tup
        return torch.from_numpy(np.expand_dims(np.expand_dims(im.cpu().detach().numpy()[r:r+tilesize, c:c+tilesize],0),0)).cuda()

    return [get_tile(tup) for tup in tile_pos]



def untile_image(imlist, imshape):
    """
    Puts together tiles into an image of given shape.
    """
    tilesize = imlist[0].cpu().detach().numpy().shape[2]
    offset =(tilesize)//2

    # Get starting and stopping index along direction
    def get_start_stop(ix, n):
        if ix==0:
            start = 0
        else:
            start = offset // 2
        if ix+tilesize==n:
            stop = tilesize
        else:
            stop = tilesize - offset // 2
        return start, stop

    (nrows, ncols) = imshape
    print(nrows,ncols)
    tile_pos = get_tile_pos(imshape, tilesize)
    out_im = np.zeros(imshape)
    for im, tup in zip(imlist, tile_pos):
        (r, c) = tup
        rsta, rsto = get_start_stop(r, nrows)
        csta, csto = get_start_stop(c, ncols)
        print(c,csta,csto)
        out_im[r+rsta:r+rsto, c+csta:c+csto] = im.cpu().detach().numpy()[0,0,rsta:rsto,csta:csto]

    return out_im
