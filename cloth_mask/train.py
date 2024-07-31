import numpy as np 
import os
import torch
import torch.nn as nn
# import torchmetrics
# from torchmetrics import Dice, JaccardIndex
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import AverageMeter, accuracy_function, train_transform, test_transform
from model import UNet
from dataset import VITONDataset


train_dataset = VITONDataset("/kaggle/input/cloth-dataset/Cloth_dataset/train", "/kaggle/input/cloth-dataset/Cloth_dataset/train.txt", train_transform)
test_dataset = VITONDataset("/kaggle/input/cloth-dataset/Cloth_dataset/test", "/kaggle/input/cloth-dataset/Cloth_dataset/test.txt", test_transform)

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load data
batch_size = 16
n_workers = os.cpu_count()
print("num_workers =", n_workers)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=n_workers)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=n_workers)

#model
model = UNet(1).to(device)

#loss
criterion = nn.BCEWithLogitsLoss()

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
n_eps = 30

#metrics
# dice_fn = torchmetrics.Dice(num_classes=2, average="macro").to(device)
# iou_fn = torchmetrics.JaccardIndex(num_classes=2, task="binary", average="macro").to(device)

#meter
acc_meter = AverageMeter()
train_loss_meter = AverageMeter()
dice_meter = AverageMeter()
iou_meter = AverageMeter()



for ep in range(1, 1+n_eps):
    acc_meter.reset()
    train_loss_meter.reset()
    dice_meter.reset()
    iou_meter.reset()
    model.train()

    for batch_id, (x, y) in enumerate(tqdm(trainloader), start=1):
        optimizer.zero_grad()
        n = x.shape[0]
        x = x.to(device).float()
        y = y.to(device).float()
        y = y / 255.0
        y_hat = model(x)
        y_hat = y_hat.squeeze() # -> logit (-vc, +vc)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_hat_mask = y_hat.sigmoid().round().long() # -> mask (0, 1)
            # dice_score = dice_fn(y_hat_mask, y.long())
            # iou_score = iou_fn(y_hat_mask, y.long())
            accuracy = accuracy_function(y_hat_mask, y.long())

            train_loss_meter.update(loss.item(), n)
            # iou_meter.update(iou_score.item(), n)
            # dice_meter.update(dice_score.item(), n)
            acc_meter.update(accuracy.item(), n)

    print("EP {}, train loss = {}, accuracy = {}, IoU = {}, dice = {}".format(
        ep, train_loss_meter.avg, acc_meter.avg, iou_meter.avg, dice_meter.avg
    ))
    if ep >= 25:
        torch.save(model.state_dict(), "/kaggle/working/model_ep_{}.pth".format(ep))