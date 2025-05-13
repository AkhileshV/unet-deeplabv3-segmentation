#!/usr/bin/env python3.11
# from operator import concat
import os
import glob
import json
import datetime

import cv2
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models

from models import Unet

# '''
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=3, progress=True)
model.classifier = nn.Conv2d(2048, 3, kernel_size=1)
model = nn.DataParallel(model)
model.to(device)
checkpoint = torch.load("/path/to/best/model.pth", map_location=device)
model.load_state_dict(checkpoint['model'])
total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params}, Trainable: {train_params}")
i=0
for img_path in glob.glob("test_10/*.jpg"):
    orig_image = cv2.imread(img_path)
    resize_image = cv2.resize(orig_image, (512, 512))
    image = np.float32(resize_image) / 127.5 - 1
    image_tensor = torch.from_numpy(image)
    image_tensor = image_tensor.permute(2,0,1)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    output = model(image_tensor)['out'].squeeze(0).permute(1,2,0)
    output = output.detach().cpu().numpy()
    output = output.astype(np.uint8)
    concat = cv2.hconcat([resize_image, (output*255).astype(np.uint8)]) 
    # cv2.imwrite(f"outputs/deeplabv3_{i}.png", concat)
    cv2.imshow("image", resize_image)
    # cv2.imshow('outpt0', output[:, :, 0])
    # cv2.imshow('outpt1', output[:, :, 1])
    cv2.imshow('outpt', output)
    cv2.waitKey()
    i+=1


