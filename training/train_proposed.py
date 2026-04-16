import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.proposed_models import ProposedModel
from utils.dataset_loader import DeepfakeDataset
from utils.fft_features import fft_feature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = DeepfakeDataset(os.path.join(PROJECT_ROOT, "Dataset", "train"))
loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = ProposedModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
accuracy_list = []

for epoch in range(10):
    model.train()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        freq = []
        for img in images:
            img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
            freq.append(fft_feature(img_np))

        freq = torch.from_numpy(np.stack(freq)).unsqueeze(1).float().to(device)
        outputs = model(images, freq)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    accuracy_list.append(acc)
    print("Epoch", epoch, "Accuracy", acc)

os.makedirs(os.path.join(PROJECT_ROOT, "saved_models"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "saved_models", "proposed_model.pth"))
plt.plot(accuracy_list)
plt.savefig(os.path.join(PROJECT_ROOT, "results", "proposed_accuracy.png"))
