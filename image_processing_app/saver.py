import os
import cv2
from datetime import datetime

BASE_DIR = "output"

def save_image(image, operation_name):
    os.makedirs(BASE_DIR, exist_ok=True)

    folder = os.path.join(BASE_DIR, operation_name)
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{operation_name}_{timestamp}.png"
    path = os.path.join(folder, filename)

    cv2.imwrite(path, image)
