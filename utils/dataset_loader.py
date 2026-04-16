import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):

    def __init__(self, path, augment=False, robust_aug=False, domain_randomization=False):

        self.images=[]
        self.labels=[]
        self.augment = augment
        self.robust_aug = robust_aug
        self.domain_randomization = domain_randomization

        for label in ["real","fake"]:

            folder=os.path.join(path,label)

            for img in os.listdir(folder):

                img_path=os.path.join(folder,img)

                self.images.append(img_path)

                if label=="real":
                    self.labels.append(0)
                else:
                    self.labels.append(1)

    def __len__(self):

        return len(self.images)

    def _jpeg_compress(self, img):
        quality = np.random.randint(40, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, enc = cv2.imencode(".jpg", img, encode_param)
        if not ok:
            return img
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec if dec is not None else img

    def _apply_augmentations(self, img):
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)

        # Color jitter (brightness/contrast).
        alpha = np.random.uniform(0.8, 1.2)  # contrast
        beta = np.random.uniform(-20, 20)    # brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if self.robust_aug:
            if np.random.rand() < 0.3:
                k = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)
            if np.random.rand() < 0.3:
                img = self._jpeg_compress(img)
            if np.random.rand() < 0.3:
                noise = np.random.normal(0, 10, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if self.domain_randomization:
            gamma = np.random.uniform(0.8, 1.2)
            img = np.clip(((img.astype(np.float32) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
            channel_scale = np.random.uniform(0.9, 1.1, size=(1, 1, 3)).astype(np.float32)
            img = np.clip(img.astype(np.float32) * channel_scale, 0, 255).astype(np.uint8)

        return img

    def __getitem__(self,idx):

        img=cv2.imread(self.images[idx])

        img=cv2.resize(img,(224,224))

        if self.augment:
            img = self._apply_augmentations(img)

        img=img/255.0

        img=np.transpose(img,(2,0,1))

        return torch.tensor(img,dtype=torch.float32), torch.tensor(self.labels[idx])
