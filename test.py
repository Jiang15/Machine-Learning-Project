from skimage import io
from util_pred_cpu import *
from skimage.morphology import dilation, disk
from skimage.external import tifffile


def predict(im, net):
    """
    Takes entire image, breakes it into tiles, calculates
    predictions, puts it back together,
    returns predictions for all pixels of the orinal image.
    """
    tile_size = 512
    if np.min([im.shape[2], im.shape[3]]) < 512:
        tile_size = np.min(im.cpu().shape[2], im.shape[3])

    tile_list = tile_image(im[0][0], tile_size)
    tol = []
    for t in tile_list:
        with torch.no_grad():
            pre = net(t)
        tol.append(pre)
    out = untile_image(tol, im[0][0].shape)
    out = np.expand_dims(np.expand_dims(out, 0), 0)
    return torch.from_numpy(out)


device = torch.device('cpu')
model = torch.load('model.pkl', map_location=device)
mask = []
for i in np.arange(16):
    img = io.imread('./testset/test' + str(i) + '.tiff')
    img = img.astype('float32')
    img = torch.from_numpy(np.expand_dims(np.expand_dims(img, 0), 0))
    img = img.to(device='cpu', dtype=torch.float32)
    pred = predict(img, model)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    stru = disk(1)
    pred = dilation(pred[0][0], stru)
    pred = torch.from_numpy(np.expand_dims(pred, 0))
    # pred = torch.from_numpy(pred.numpy()[0])
    img_seg = segment(pred)
    mask.append(img_seg.numpy()[0][0].astype('int16'))
tifffile.imsave('test_mask.tif', np.array(mask), imagej=True)
