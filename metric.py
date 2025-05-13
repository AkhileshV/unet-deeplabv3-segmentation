import glob

import cv2
import numpy as np
from sklearn.metrics import jaccard_score, f1_score
import torch
import torch.nn as nn
from torchvision import models

from models import Unet


if __name__ == "__main__":
    train_images = glob.glob("/path/to/sample/train/images/*.jpg")
    val_images = glob.glob("/path/to/sample/val/images/*.jpg")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold = 0.5
    print(device)

    # initialize and load UNet unet
    unet = Unet()
    unet = nn.DataParallel(unet)
    unet.to(device)
    checkpoint = torch.load("/path/to/best/unet/model.pth", map_location=device)
    unet.load_state_dict(checkpoint['model'])

    # initialize and load deeplabv3 model
    model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=3, progress=True)
    model.classifier = nn.Conv2d(2048, 3, kernel_size=1)
    model = nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load("/path/to/best/deeplabv3/model.pth", map_location=device)
    model.load_state_dict(checkpoint['model'])

    train_unet, val_unet = [], []
    train_deeplab, val_deeplab = [], []

    # train loop
    for img_path in train_images:

        # preprocess image
        orig_image = cv2.imread(img_path)
        resize_image = cv2.resize(orig_image, (512, 512))
        image = np.float32(resize_image) / 127.5 - 1
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(2,0,1)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # preprocess label mask
        label_mask_path = img_path.replace("images", "masks").replace(".jpg", " 2.png")
        label_mask = cv2.imread(label_mask_path)
        label_mask = cv2.resize(label_mask, (512, 512))[:, :, 0]
        label_mask = label_mask.flatten()

        # obtain unet output and iou score
        output = unet(image_tensor).squeeze(0).permute(1,2,0)
        output = output.detach().cpu().numpy()
        mask_channel_1 = (output[:, :, 0] > threshold).astype(np.uint8)
        mask_channel_2 = (output[:, :, 1] > threshold).astype(np.uint8)
        mask_channel_3 = (output[:, :, 2] > threshold).astype(np.uint8)
        merged_mask = mask_channel_1 + 2 * mask_channel_2 + 3 * mask_channel_3
        merged_mask = (merged_mask - 1).flatten()

        unet_miou = f1_score(label_mask, merged_mask, average='weighted')
        if unet_miou > 0.5:
            train_unet.append(unet_miou)

        # obtain deeplab output and iou score
        output = model(image_tensor)['out'].squeeze(0).permute(1,2,0)
        output = output.detach().cpu().numpy()
        mask_channel_1 = (output[:, :, 0] > threshold).astype(np.uint8)
        mask_channel_2 = (output[:, :, 1] > threshold).astype(np.uint8)
        mask_channel_3 = (output[:, :, 2] > threshold).astype(np.uint8)
        merged_mask = mask_channel_1 + 2 * mask_channel_2 + 3 * mask_channel_3
        merged_mask = (merged_mask - 1).flatten()

        deeplab_miou = f1_score(label_mask, merged_mask, average='weighted')
        if deeplab_miou > 0.5:
            train_deeplab.append(deeplab_miou)

    # val loop
    for img_path in val_images:

        # preprocess image
        orig_image = cv2.imread(img_path)
        resize_image = cv2.resize(orig_image, (512, 512))
        image = np.float32(resize_image) / 127.5 - 1
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(2,0,1)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # preprocess label mask
        label_mask_path = img_path.replace("images", "masks").replace(".jpg", " 2.png")
        label_mask = cv2.imread(label_mask_path)
        label_mask = cv2.resize(label_mask, (512, 512))[:, :, 0]
        label_mask = label_mask.flatten()

        # obtain unet output and iou score
        output = unet(image_tensor).squeeze(0).permute(1,2,0)
        output = output.detach().cpu().numpy()
        mask_channel_1 = (output[:, :, 0] > threshold).astype(np.uint8)
        mask_channel_2 = (output[:, :, 1] > threshold).astype(np.uint8)
        mask_channel_3 = (output[:, :, 2] > threshold).astype(np.uint8)
        merged_mask = mask_channel_1 + 2 * mask_channel_2 + 3 * mask_channel_3
        merged_mask = (merged_mask - 1).flatten()

        unet_miou = f1_score(label_mask, merged_mask, average='weighted')
        if unet_miou > 0.5:
            val_unet.append(unet_miou)

        # obtain deeplab output and iou score
        output = model(image_tensor)['out'].squeeze(0).permute(1,2,0)
        output = output.detach().cpu().numpy()
        mask_channel_1 = (output[:, :, 0] > threshold).astype(np.uint8)
        mask_channel_2 = (output[:, :, 1] > threshold).astype(np.uint8)
        mask_channel_3 = (output[:, :, 2] > threshold).astype(np.uint8)
        merged_mask = mask_channel_1 + 2 * mask_channel_2 + 3 * mask_channel_3
        merged_mask = (merged_mask - 1).flatten()

        deeplab_miou = f1_score(label_mask, merged_mask, average='weighted')
        if deeplab_miou > 0.5:
            val_deeplab.append(deeplab_miou)

    print(f"UNet Train mIOU: {sum(train_unet)/len(train_unet)}, Val mIOU: {sum(val_unet)/len(val_unet)}")
    print(f"DeepLabv3 Train mIOU: {sum(train_deeplab)/len(train_deeplab)}, Val mIOU: {sum(val_deeplab)/len(val_deeplab)}")
