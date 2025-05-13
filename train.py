#!/usr/bin/env python3.11
import os
import glob
import json
import datetime

import cv2
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from preprocess import DrivableDataset
from models import Unet
from metric import compute_miou

def load_checkpoint(checkpoint_path, device, model, optimizer):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    curr_epoch = checkpoint['epoch']
    return model, optimizer, curr_epoch

date_time = datetime.datetime.now()
formatted_datetime = date_time.strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(formatted_datetime, exist_ok=True)

is_pretrained_available = True
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "/path/to/your/dataset/"
train_images = glob.glob(data_dir + "train/*.jpg")
val_images = glob.glob(data_dir + "val/*.jpg")

trainset = DrivableDataset(train_images)
valset = DrivableDataset(val_images)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)

model = Unet()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
curr_epoch=0

if is_pretrained_available:
    model, optimizer, curr_epoch = load_checkpoint("best_model.pth", device, model, optimizer)

num_epochs = 10
train_losses, val_losses = [], []
min_val_loss = np.Inf
log_dict = {}
for epoch in range(curr_epoch, num_epochs):
  # training loop
    model.train()
    running_loss = 0.0
    train_iou = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iou = compute_miou(outputs, labels)
        train_iou += iou

    train_losses.append(running_loss/len(trainloader))

    # validation loop
    model.eval()
    val_running_loss = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            iou = compute_miou(outputs, labels)
            val_iou += iou

    val_losses.append(val_running_loss/len(valloader))

    if val_running_loss/len(valloader) < min_val_loss:
        min_val_loss = val_running_loss/len(valloader)
        checkpoint = {"model": model.state_dict(), "optimizer":optimizer.state_dict(), "epoch":epoch}
        torch.save(formatted_datetime + "/best_model.pth", checkpoint)
        print("Saving Checkpoint to", formatted_datetime, "/best_model.pth")
    
    log_dict[epoch] = {'train_iou': train_iou, 'val_iou': val_iou, 'epoch':epoch, 'train_loss':running_loss/len(trainloader), 'val_loss':val_running_loss/len(valloader)}
    with open(formatted_datetime + "/log.json", 'w+') as f:
        json.dump(f, log_dict)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(trainloader)}, Val Loss: {val_running_loss/len(valloader)}")
    print(f"Epoch {epoch+1}/{num_epochs}, Train IoU: {train_iou/len(trainloader)}, Val IoU: {val_iou/len(valloader)}")

