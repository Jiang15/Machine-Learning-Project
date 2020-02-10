from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img):
        w, h = pil_img.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        pil_img = pil_img.resize((newW, newH))
        img_nd = np.array(pil_img)
        img_nd = np.expand_dims(img_nd, axis=0)
        return img_nd

    def __getitem__(self, i):
        idx = self.ids[i]
        print(idx)
        idy = idx.replace('frame', 'mask')
        mask_file = glob(self.masks_dir + idy + '.tiff')
        img_file = glob(self.imgs_dir + idx + '.tiff')
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        img = self.preprocess(img).astype(float)
        mask = self.preprocess(mask).astype(int)
        binary_mask = mask.copy()
        binary_mask[binary_mask > 0] = 1
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask),
                'binary_mask': torch.from_numpy(binary_mask)}
