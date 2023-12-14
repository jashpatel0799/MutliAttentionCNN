# import libs
import torch
from torch import nn
import cv2

import torchvision
from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAccuracy

from adv_data import add_adverasial_noise
from train import train_loop, test_loop, plotplot
from model import ResNet50Attention, loss_fn

# Device Agnostic Code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# LoadData
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(root='/iitjhome/m22cs061/DL_Project/CUB_200_2011/images', transform=transform)

# plt.imshow(train_dataset[0][0].permute(1,2,0))
# plt.axis(False)

class_name = train_dataset.classes
num_class = len(class_name)

train_size = int(0.75 * len(train_dataset))
test_size = len(train_dataset) - train_size
trainset, testset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

BATCH_SIZE = 32
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


# import torch.nn.functional as F
class Attention(torch.nn.Module):
    """
    Attention block for CNN model.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Attention, self).__init__()
        self.conv_depth = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=in_channels)
        self.conv_point = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = torch.nn.Tanh()
    def forward(self, inputs):
        x, output_size = inputs
        x = F.adaptive_max_pool2d(x, output_size=output_size)
        print(f"conv_x : {x.shape}")
        x = self.conv_depth(x)
        x = self.conv_point(x)
        x = self.bn(x)
        x = self.activation(x) + 1.0
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        avg_out = self.avgpool(out)
        max_out = self.maxpool(out)
        out = avg_out + max_out
        out = self.sigmoid(out)
        return x * out

class ResNet50Attention(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50Attention, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=False, progress = False)
        self.attention_block1 = AttentionBlock(256)
        self.attention_block2 = AttentionBlock(512)
        self.attention_block3 = AttentionBlock(1024)
        self.attention_block4 = AttentionBlock(2048)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.attention_block1(x)

        x = self.resnet.layer2(x)
        x = self.attention_block2(x)

        x = self.resnet.layer3(x)
        x = self.attention_block3(x)

        x = self.resnet.layer4(x)
        x = self.attention_block4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Instantiate the ResNet50Attention model

class Attention_GradCAM:
    def __init__(self, model, feature_layer):
        self.model = model
        self.feature_layer = feature_layer
        self.gradient = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn_forward(module, input, output):
            self.activations = output

        def hook_fn_backward(module, grad_out, grad_in):
            self.gradient = grad_out[0]

        for name, module in self.model.named_modules():
            if name == self.feature_layer:
                self.hooks.append(module.register_forward_hook(hook_fn_forward))
                self.hooks.append(module.register_backward_hook(hook_fn_backward))

    def _get_weights(self):
      if self.gradient is None:
          raise ValueError("Gradient is None. Backward hook did not execute.")
      return torch.mean(self.gradient, dim=(2, 3), keepdim=True)

    def generate(self, input_image, class_idx=None):
        output = self.model(input_image)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        self.model.zero_grad()
        output[0][class_idx].backward()
        weights = self._get_weights()
        cam = torch.sum(weights * self.activations, axis=1, keepdim=True)
        cam = F.relu(cam)
        cam /= torch.max(cam)
        return cam.squeeze().detach().cpu().numpy()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

att_resnet50 = ResNet50Attention()

images = [5, 10, 15, 20, 25]

plt.figure(figsize=(12,5))
row, col = 1, len(images)

for i in range(1, row*col+1):
  image, label = train_dataset[images[i-1]]
  plt.subplot(row, col, i)
  plt.imshow(image.permute(1, 2, 0))
  # plt.title(label)
  plt.axis(False)

plt.savefig("originals.jpg")

# Load the input image and pre-process it
for i in range(len(images)):
  # random_idx = torch.randint(0, len(train_dataset), size = [1]).item()
  data_image = train_dataset[images[i]][0]
  input_image = data_image.unsqueeze(0)
  # print(input_image.shape)
  # print(input_image.dtype)

  data_image = data_image.to(dtype = torch.float64)
  image = data_image.cpu().permute(1, 2, 0).detach().numpy() * 255

  # print(image.shape)
  # Initialize Grad-CAM and generate the heatmap
  attention_gradcam = Attention_GradCAM(att_resnet50, 'attention_block4')
  heatmap = attention_gradcam.generate(input_image)

  # Resize the heatmap to match the input image size
  heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

  heatmap = heatmap.astype(float)
  image = image.astype(float)

  # Overlay the heatmap onto the input image
  output_ma_resnet50 = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

  # plt.imshow(output_ma_resnet50/255)
  # Save the output image
  cv2.imwrite(f"{'output_ma_resnet50_'+str(images[i])+'.jpg'}", output_ma_resnet50)