import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, roc_curve
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.baseline_models import BaselineCNN
from models.proposed_models import ProposedModel
from utils.dataset_loader import DeepfakeDataset
from utils.fft_features import fft_feature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = os.path.join(PROJECT_ROOT, "results")
os.makedirs(results_dir, exist_ok=True)

test_dataset = DeepfakeDataset(os.path.join(PROJECT_ROOT, "Dataset", "test"))
loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

baseline = BaselineCNN(width_mult=0.75, use_mixstyle=True).to(device)
baseline.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "saved_models", "baseline_model.pth"), map_location=device))
baseline.eval()

proposed = ProposedModel().to(device)
proposed.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "saved_models", "proposed_model.pth"), map_location=device))
proposed.eval()

y_true = []
y_pred_base = []
y_pred_prop = []
y_scores = []
latency_base_total = 0.0
latency_prop_total = 0.0
sample_count = 0

y_pred_base_noisy = []
y_pred_prop_noisy = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        sample_count += batch_size

        start = time.perf_counter()
        out1 = baseline(images)
        latency_base_total += time.perf_counter() - start
        pred1 = torch.argmax(out1, dim=1)
        y_pred_base.extend(pred1.detach().cpu().numpy())

        freq = []
        for img in images:
            img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
            freq.append(fft_feature(img_np))

        freq = torch.from_numpy(np.stack(freq)).unsqueeze(1).float().to(device)
        start = time.perf_counter()
        out2 = proposed(images, freq)
        latency_prop_total += time.perf_counter() - start
        pred2 = torch.argmax(out2, dim=1)
        probs = torch.softmax(out2, dim=1)[:, 1]

        y_scores.extend(probs.detach().cpu().numpy())
        y_pred_prop.extend(pred2.detach().cpu().numpy())
        y_true.extend(labels.detach().cpu().numpy())

        # Robustness check on noisy inputs.
        noisy_images = torch.clamp(images + 0.05 * torch.randn_like(images), 0.0, 1.0)

        out1_noisy = baseline(noisy_images)
        pred1_noisy = torch.argmax(out1_noisy, dim=1)
        y_pred_base_noisy.extend(pred1_noisy.detach().cpu().numpy())

        freq_noisy = []
        for img in noisy_images:
            img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
            freq_noisy.append(fft_feature(img_np))

        freq_noisy = torch.from_numpy(np.stack(freq_noisy)).unsqueeze(1).float().to(device)
        out2_noisy = proposed(noisy_images, freq_noisy)
        pred2_noisy = torch.argmax(out2_noisy, dim=1)
        y_pred_prop_noisy.extend(pred2_noisy.detach().cpu().numpy())

baseline_acc = accuracy_score(y_true, y_pred_base)
proposed_acc = accuracy_score(y_true, y_pred_prop)
baseline_f1 = f1_score(y_true, y_pred_base)
proposed_f1 = f1_score(y_true, y_pred_prop)
baseline_noisy_acc = accuracy_score(y_true, y_pred_base_noisy)
proposed_noisy_acc = accuracy_score(y_true, y_pred_prop_noisy)

baseline_latency_ms = (latency_base_total / sample_count) * 1000 if sample_count else 0.0
proposed_latency_ms = (latency_prop_total / sample_count) * 1000 if sample_count else 0.0
baseline_robustness = (baseline_noisy_acc / baseline_acc) * 100 if baseline_acc > 0 else 0.0
proposed_robustness = (proposed_noisy_acc / proposed_acc) * 100 if proposed_acc > 0 else 0.0

print("\nBaseline Model Metrics")
print("Accuracy:", baseline_acc)
print("F1 Score:", baseline_f1)
print("Inference Latency (ms/sample):", baseline_latency_ms)
print("Robustness Score (% acc retention):", baseline_robustness)

print("\nProposed Model Metrics")
print("Accuracy:", proposed_acc)
print("F1 Score:", proposed_f1)
print("Inference Latency (ms/sample):", proposed_latency_ms)
print("Robustness Score (% acc retention):", proposed_robustness)

cm = confusion_matrix(y_true, y_pred_prop)
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.clf()

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label="AUC=" + str(roc_auc))
plt.legend()
plt.title("ROC Curve")
plt.savefig(os.path.join(results_dir, "roc_curve.png"))
plt.clf()

plt.bar(["Baseline", "Proposed"], [baseline_acc, proposed_acc])
plt.title("Model Comparison")
plt.savefig(os.path.join(results_dir, "comparison_plot.png"))
