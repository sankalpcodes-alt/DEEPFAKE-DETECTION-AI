import time

import torch
from sklearn.metrics import accuracy_score, f1_score


def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def calculate_f1(y_true, y_pred, average="binary"):
    return f1_score(y_true, y_pred, average=average)


def calculate_metrics(y_true, y_pred, average="binary"):
    accuracy = calculate_accuracy(y_true, y_pred)
    f1 = calculate_f1(y_true, y_pred, average=average)
    return accuracy, f1


def calculate_inference_latency(model, dataloader, device="cpu", warmup_batches=1, max_batches=None):
    model = model.to(device)
    model.eval()

    processed_batches = 0
    processed_samples = 0
    elapsed = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and processed_batches >= max_batches:
                break

            images = batch[0].to(device)

            if batch_idx < warmup_batches:
                _ = model(images)
                continue

            start = time.perf_counter()
            _ = model(images)
            elapsed += time.perf_counter() - start

            processed_batches += 1
            processed_samples += images.size(0)

    if processed_batches == 0 or processed_samples == 0:
        return 0.0, 0.0

    ms_per_batch = (elapsed / processed_batches) * 1000.0
    ms_per_sample = (elapsed / processed_samples) * 1000.0
    return ms_per_batch, ms_per_sample


def calculate_robustness_score(clean_accuracy, perturbed_accuracy):
    if clean_accuracy <= 0:
        return 0.0
    return perturbed_accuracy / clean_accuracy
