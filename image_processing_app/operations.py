import cv2
import numpy as np

# ---------- Image Enhancement ----------

def histogram_equalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def smooth_gaussian(img, ksize=5):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def sharpen_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharp = gray - lap
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

# ---------- Noise & Restoration ----------

def add_gaussian_noise(img, level=20):
    noise = np.random.normal(0, level, img.shape).astype(np.int16)
    noisy = img.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img, prob=0.02):
    noisy = img.copy()
    rnd = np.random.rand(*img.shape[:2])
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 255
    return noisy

def median_filter(img, ksize=5):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)

# ---------- Color Processing ----------

def to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def color_enhancement(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# ---------- Frequency Domain ----------

def low_pass_filter(img, cutoff=30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fftshift(np.fft.fft2(gray))
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1

    f_filtered = f * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
    return cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def high_pass_filter(img, cutoff=30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fftshift(np.fft.fft2(gray))
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), np.uint8)
    mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 0

    f_filtered = f * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
    return cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2BGR)
