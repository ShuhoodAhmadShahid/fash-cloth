import sys
try:
    import mediapipe
    print("Mediapipe: OK")
except ImportError:
    print("Mediapipe: MISSING")

try:
    import transformers
    print("Transformers: OK")
except ImportError:
    print("Transformers: MISSING")

try:
    import cv2
    print("OpenCV: OK")
except ImportError:
    print("OpenCV: MISSING")

try:
    import torch
    print("Torch: OK")
except ImportError:
    print("Torch: MISSING")
