import logging
from tqdm import tqdm
from util_pred import *
from metric import quality_measures



def eval_net(net, loader, device, n_val):
    net.eval()
    res = []
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            mask_pred = predict_test(imgs, net)
            for true_mask, pred in zip(true_masks, mask_pred):
                mask_thre = threshold(pred, 0.5)
                mask_seg = segment(mask_thre)
                res = quality_measures(true_mask.cpu().detach().numpy()[0], mask_seg.detach().numpy()[0][0])
                logging.info('Quality measure: {}'.format(res))
            pbar.update(imgs.shape[0])

    return res
