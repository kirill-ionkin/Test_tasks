import os
import sys


import numpy as np

import torch

import torchvision
from torchvision import transforms
from torchvision import models

from PIL import Image


ROOT= ""
MODEL_SAVE = "model_checkpoints"
resnet18_checkpoint = "custom_resnet18.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

resnet18 = models.resnet18()
resnet18.fc = torch.nn.Linear(in_features=512, out_features=2)
resnet18.load_state_dict(torch.load(os.path.join(MODEL_SAVE, resnet18_checkpoint)))


class OpenEyesClassificator():
	def __init__(self, model=resnet18, device=device, transforms=transforms_rgb):
		self.device = device
		self.model = model.to(device)
		self.transforms = transforms

	def predict(self, inpIm):
		pil_img = self.open_img(inpIm)
		x = self.preprocess_img(pil_img)
		x = x.to(self.device)

		self.model.eval()
		with torch.no_grad():
			outputs = self.model(x)
		return round(torch.nn.functional.softmax(outputs.cpu(), dim=1)[:, 1].item(), 4)

	def open_img(self, inpIm):
		return Image.open(inpIm).convert("RGB")

	def preprocess_img(self, pil_img):
		x = self.transforms(pil_img)
		return torch.unsqueeze(x, 0)


if __name__ == "__main__":
	inpIm = sys.argv[1]

	classificator = OpenEyesClassificator()
	preds = classificator.predict(inpIm)
	print(preds)
