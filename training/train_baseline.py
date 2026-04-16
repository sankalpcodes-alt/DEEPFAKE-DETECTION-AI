import os
import sys
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataset_loader import DeepfakeDataset
from models.baseline_models import BaselineCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EFFICIENCY_ALPHA = 0.002  # Lower favors accuracy more; higher favors latency more.

train_dataset=DeepfakeDataset(
    os.path.join(PROJECT_ROOT, "Dataset", "train"),
    augment=True,
    robust_aug=True,
    domain_randomization=True,
)

loader=DataLoader(train_dataset,batch_size=16,shuffle=True)

model=BaselineCNN(width_mult=0.75, use_mixstyle=True).to(device)
criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

accuracy_list=[]
loss_list=[]
latency_list=[]
tradeoff_list=[]
best_tradeoff = float("-inf")

for epoch in range(10):

    model.train()
    correct=0
    total=0
    running_loss=0
    epoch_latency_s = 0.0

    for images,labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs=model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_latency_s += (time.perf_counter() - t0)

        loss=criterion(outputs,labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()

        preds=torch.argmax(outputs,1)

        correct+=(preds==labels).sum().item()

        total+=labels.size(0)

    acc=correct/total
    avg_latency_ms = (epoch_latency_s / total) * 1000 if total > 0 else 0.0
    tradeoff_score = acc - (EFFICIENCY_ALPHA * avg_latency_ms)

    accuracy_list.append(acc)
    loss_list.append(running_loss)
    latency_list.append(avg_latency_ms)
    tradeoff_list.append(tradeoff_score)

    if tradeoff_score > best_tradeoff:
        best_tradeoff = tradeoff_score
        os.makedirs(os.path.join(PROJECT_ROOT, "saved_models"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "saved_models", "baseline_model.pth"))

    print("Epoch", epoch, "Accuracy", acc)

os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

plt.plot(accuracy_list)
plt.savefig(os.path.join(PROJECT_ROOT, "results", "baseline_accuracy.png"))
plt.clf()
plt.plot(loss_list)
plt.savefig(os.path.join(PROJECT_ROOT, "results", "loss_plot.png"))
