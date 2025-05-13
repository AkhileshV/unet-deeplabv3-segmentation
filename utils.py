import json
import matplotlib.pyplot as plt

# DeepLab V3 model
with open("/path/to/deeplab/train/logs.json") as f:
    log_data = json.load(f)

train_loss = []
val_loss = []

for epoch in log_data:
    train_loss.append(log_data[epoch]['train_loss'])
    val_loss.append(log_data[epoch]['val_loss'])

epochs = list(range(len(train_loss)))
plt.plot(epochs, train_loss, color='r', label='Train Loss')
plt.plot(epochs, val_loss, color='g', label='Val Loss')
plt.legend(loc='upper right')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.title('Loss vs Epoch graph - DeepLabv3')
plt.savefig("outputs/loss-graph-deeplab.png")
plt.show()

# plotting graph for UNet model
# train_loss = [0.2489, 0.1732, 0.1593, 0.1528, 0.1485, 0.1461, 0.1444, 0.1426, 0.1248, 0.1207]
# val_loss = [0.2110, 0.1832, 0.1638, 0.1505, 0.1731, 0.1558, 0.1536, 0.1537, 0.1261, 0.1268]
with open("/path/to/unet/train/logs.json") as f:
    log_data = json.load(f)

train_loss = []
val_loss = []

for epoch in log_data:
    train_loss.append(log_data[epoch]['train_loss'])
    val_loss.append(log_data[epoch]['val_loss'])
epochs = list(range(len(train_loss)))
plt.plot(epochs, train_loss, color='r', label='Train Loss')
plt.plot(epochs, val_loss, color='g', label='Val Loss')
plt.legend(loc='upper right')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.title('Loss vs Epoch graph - UNet')
plt.savefig("loss-graph-unet.png")
plt.show()

