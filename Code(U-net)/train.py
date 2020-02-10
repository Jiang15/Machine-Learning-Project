import argparse
import logging
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from validation import eval_net
from unet import UNet
from tensorboardX import SummaryWriter
from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from util_pred import *

dir_img = './frame/'
dir_mask = './mask/'


def train_net(net, device, epochs=3, batch_size=1, lr=0.1, val_percent=0.1, save_cp=True, img_scale=1):
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    for epoch in range(epochs):
        net.train()
        step = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch["image"]
                true_masks = batch["binary_mask"]
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                tiles_img, tile_mask = predict_train(imgs, true_masks, 512)
                loss_sum = 0
                cnt = 0
                for i, m in zip(tiles_img, tile_mask):
                    if np.sum(m.cpu().detach().numpy()) > 36:
                        criterion = nn.BCEWithLogitsLoss()
                        cnt += 1
                        mask_pred = net(i)

                        loss = criterion(mask_pred, m)
                        loss_sum += loss.item()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                writer.add_scalar('Loss/train', loss_sum / cnt, global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])
                step += 1

        val_score = eval_net(net, val_loader, device, n_val)

    writer.close()

    return net


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=5,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # device = torch.device('cpu')
    logging.info(f'Using device {device}')
    n_net = UNet(n_channels=1, n_classes=1)
    logging.info(f'Network:\n'
                 f'\t{n_net.n_channels} input channels\n'
                 f'\t{n_net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if n_net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        n_net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    n_net.to(device=device)

    model = train_net(net=n_net, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr,
                      device=device, img_scale=args.scale, val_percent=args.val / 100)

    torch.save(model, 'model.pkl')
