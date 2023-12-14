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


# Define the Grad-CAM function
class GradCAM:
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
        return torch.mean(self.gradient, axis=(2, 3), keepdims=True)

    def generate(self, input_image, class_idx=None):
        output = self.model(input_image)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        self.model.zero_grad()
        output[0][class_idx].backward()
        weights = self._get_weights()
        cam = torch.sum(weights * self.activations, axis=1, keepdims=True)
        cam = F.relu(cam)
        cam /= torch.max(cam)
        return cam.squeeze().detach().cpu().numpy()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

model = models.resnet50(pretrained=False, progress = False)

images = [5, 10, 15, 20, 25]
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
  gradcam = GradCAM(model, 'layer4')
  heatmap = gradcam.generate(input_image)
  # print(f"1: {heatmap.shape}")

  # Resize the heatmap to match the input image size
  heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

  # print(f"2: {heatmap.shape}")

  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


  # print(image)
  # print(heatmap)
  # print(image.shape, type(image), image.dtype)
  # print(heatmap.shape, type(heatmap), heatmap.dtype)
  # print()
  heatmap = heatmap.astype(float)
  # image = image.astype(float)


  # Overlay the heatmap onto the input image
  output_resnet50 = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
  # Save the output image
  # plt.imshow(output_resnet50/255)
  cv2.imwrite(f"{'output_resnet50_'+str(images[i])+'.jpg'}", output_resnet50)