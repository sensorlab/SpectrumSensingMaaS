from typing import List
from fastapi import FastAPI
import torch
import torch.nn.functional as F
import numpy as np
import gdown
import torchvision.models as models

model = models.resnet18(num_classes=24)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

gdown.download('https://drive.google.com/uc?id=1cBPywP7OgrXFvHuEpEG-DPM9rp8wXM8n')
checkpoint = torch.load('resnet_weights.pt', map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

app = FastAPI()

@app.post("/SpectrumSensing_ResNet18")
async def echo(data: List[List[List[List[float]]]]):
    preds = []
    for i in range(len(data)):
        cur_data = torch.tensor([data[i]])
        out = model(cur_data)
        pred = out.argmax(dim=1)
        preds.append(str(pred))
    return {"preds": preds}