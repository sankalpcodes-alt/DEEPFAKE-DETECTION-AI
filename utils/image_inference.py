import os

import cv2
import numpy as np
import torch

from models.baseline_models import BaselineCNN
from models.proposed_models import ProposedModel
from utils.fft_features import fft_feature

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "baseline_model.pth")
PROPOSED_MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "proposed_model.pth")
CLASS_NAMES = {0: "Real", 1: "Fake"}

_baseline_model = None
_proposed_model = None


def _detect_primary_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        return None

    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None

    return max(faces, key=lambda face: face[2] * face[3])


def _extract_subject_crop(image):
    face = _detect_primary_face(image)
    if face is None:
        return image

    x, y, w, h = face

    # Keep the crop face-focused and remove large flat backgrounds that often skew inference.
    x1 = max(0, x + int(0.08 * w))
    y1 = max(0, y - int(0.23 * h))
    x2 = min(image.shape[1], x + int(0.95 * w))
    y2 = min(image.shape[0], y + int(0.95 * h))

    if x2 <= x1 or y2 <= y1:
        return image

    return image[y1:y2, x1:x2]


def _prepare_tensors(image):
    image = cv2.resize(image, (224, 224))
    image_float = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(np.transpose(image_float, (2, 0, 1))).unsqueeze(0).float()
    freq = fft_feature(image_float)
    freq_tensor = torch.from_numpy(freq).unsqueeze(0).unsqueeze(0).float()
    return image_tensor.to(DEVICE), freq_tensor.to(DEVICE)


def _should_use_crop_for_proposed(image, full_probabilities, crop_probabilities):
    height, width = image.shape[:2]
    portrait_like = height > (width * 1.15)

    full_fake = float(full_probabilities[1])
    crop_real = float(crop_probabilities[0])

    # Portrait images like ID photos often need face-focused inference, but square dataset frames
    # generally perform better with the original full-frame context.
    return portrait_like and full_fake > 0.90 and crop_real > 0.90


def _load_models():
    global _baseline_model, _proposed_model

    if _baseline_model is None:
        if not os.path.exists(BASELINE_MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at '{BASELINE_MODEL_PATH}'. Train the baseline model first."
            )

        baseline_model = BaselineCNN(width_mult=0.75, use_mixstyle=True).to(DEVICE)
        baseline_model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=DEVICE))
        baseline_model.eval()
        _baseline_model = baseline_model

    if _proposed_model is None:
        if not os.path.exists(PROPOSED_MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at '{PROPOSED_MODEL_PATH}'. Train the proposed model first."
            )

        proposed_model = ProposedModel().to(DEVICE)
        proposed_model.load_state_dict(torch.load(PROPOSED_MODEL_PATH, map_location=DEVICE))
        proposed_model.eval()
        _proposed_model = proposed_model

    return _baseline_model, _proposed_model


def _preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from '{image_path}'.")

    full_image_tensor, full_freq_tensor = _prepare_tensors(image)
    crop_image_tensor, crop_freq_tensor = _prepare_tensors(_extract_subject_crop(image))
    return image, full_image_tensor, full_freq_tensor, crop_image_tensor, crop_freq_tensor


def predict_image(image_path):
    baseline_model, proposed_model = _load_models()
    image, full_image_tensor, full_freq_tensor, crop_image_tensor, crop_freq_tensor = _preprocess_image(image_path)

    with torch.no_grad():
        baseline_logits = baseline_model(full_image_tensor)
        proposed_full_logits = proposed_model(full_image_tensor, full_freq_tensor)
        proposed_crop_logits = proposed_model(crop_image_tensor, crop_freq_tensor)

        baseline_probabilities = torch.softmax(baseline_logits, dim=1)[0].detach().cpu().numpy()
        proposed_full_probabilities = torch.softmax(proposed_full_logits, dim=1)[0].detach().cpu().numpy()
        proposed_crop_probabilities = torch.softmax(proposed_crop_logits, dim=1)[0].detach().cpu().numpy()

    if _should_use_crop_for_proposed(image, proposed_full_probabilities, proposed_crop_probabilities):
        proposed_probabilities = proposed_crop_probabilities
    else:
        proposed_probabilities = proposed_full_probabilities

    baseline_index = int(np.argmax(baseline_probabilities))
    proposed_index = int(np.argmax(proposed_probabilities))

    return {
        "baseline_label": CLASS_NAMES[baseline_index],
        "proposed_label": CLASS_NAMES[proposed_index],
        "baseline_real_probability": float(baseline_probabilities[0]),
        "baseline_fake_probability": float(baseline_probabilities[1]),
        "proposed_real_probability": float(proposed_probabilities[0]),
        "proposed_fake_probability": float(proposed_probabilities[1]),
    }
