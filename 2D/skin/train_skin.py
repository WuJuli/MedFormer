from __future__ import division
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from skin_data.data_loader.loader import *
import pandas as pd
import glob
import argparse
import nibabel as nib
import numpy as np
import copy
import yaml
from net.LightDCN import MedFormer
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="encoder type", required=True)
parser.add_argument("--scale", type=str, help="the scale of the encoder", required=True)
parser.add_argument('--root_path', type=str,
                    default='./skin_data/ISIC-2018_npy/', help='root dir for data')
parser.add_argument('--taskname', type=str,
                    default='isic2018', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')

parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=322, help='random seed')

parser.add_argument("--pretrain", type=str, help="using pretrained weights or not", choices=['True', 'False'],
                    required=True)
args = parser.parse_args()

## Loader
## Hyper parameters

best_val_loss = np.inf
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: {}".format(device))

train_dataset = isic_loader(path_Data=args.root_path, train=True)
train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True)
val_dataset = isic_loader(path_Data=args.root_path, train=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print("Created loaders.")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
dataset_name = args.taskname

args.exp = dataset_name + '_' + args.model + '_' + args.scale
if 'ISIC-2018' in args.root_path:
    snapshot_path = "./model_out/isic2018/{}/{}".format(args.exp, 'model')
elif 'ISIC-2017' in args.root_path:
    snapshot_path = "./model_out/isic2017/{}/{}".format(args.exp, 'model')
else:
    snapshot_path = "./model_out/ph/{}/{}".format(args.exp, 'model')
snapshot_path = snapshot_path + '_pretrainT' if args.pretrain else snapshot_path
snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                      0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.base_lr)

snapshot_path = snapshot_path + '_s' + str(args.seed)

print("Snapshot path: " + snapshot_path)

print("Batch size: {}".format(args.batch_size))

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

pretrain = args.pretrain.lower() == 'true'

net = MedFormer(model_type=args.model, model_scale=args.scale, pretrain=pretrain, num_classes=args.num_classes).to(
    device=device)

optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

criteria = torch.nn.BCEWithLogitsLoss()
print("Created net and optimizers.")
print("Start training...")

for name, param in net.named_parameters():
    if param.requires_grad:
        print(name)

for ep in range(int(args.max_epochs)):
    print("Current epoch: {}".format(ep))
    net.train()
    epoch_loss = 0
    for itter, batch in enumerate(train_loader):
        img = batch['image'].to(device, dtype=torch.float)
        # print("Image shape: {}".format(img.shape))
        msk = batch['mask'].to(device)
        mask_type = torch.float32
        msk = msk.to(device=device, dtype=mask_type)
        msk_pred = net(img)
        loss = criteria(msk_pred, msk)
        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        if itter % 20 == 0:
            print(f' Epoch>> {ep + 1} and itteration {itter + 1} Loss>> {(epoch_loss / (itter + 1))}')
    ## Validation phase
    with torch.no_grad():
        print('val_mode')
        val_loss = 0
        net.eval()
        for itter, batch in enumerate(val_loader):
            img = batch['image'].to(device, dtype=torch.float)
            msk = batch['mask'].to(device)
            mask_type = torch.float32
            msk = msk.to(device=device, dtype=mask_type)
            msk_pred = net(img)
            loss = criteria(msk_pred, msk)
            val_loss += loss.item()
        print(f' validation on epoch>> {ep + 1} dice loss>> {(abs(val_loss / (itter + 1)))}')
        mean_val_loss = (val_loss / (itter + 1))
        # Check the performance and save the model
        if mean_val_loss < best_val_loss:
            print('New best loss, saving...')
            best_val_loss = copy.deepcopy(mean_val_loss)
            state = copy.deepcopy({'model_weights': net.state_dict(), 'val_loss': best_val_loss})
            torch.save(state, snapshot_path + '/best_model.pth')

    scheduler.step(mean_val_loss)

print('Training phase finished')
