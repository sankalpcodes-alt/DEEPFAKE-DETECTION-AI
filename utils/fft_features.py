import cv2
import numpy as np

def fft_feature(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    fft = np.fft.fft2(gray)

    fft_shift = np.fft.fftshift(fft)

    magnitude = np.log(np.abs(fft_shift)+1)

    # normalize
    magnitude = magnitude / np.max(magnitude)

    magnitude = cv2.resize(magnitude,(224,224))

    return magnitude