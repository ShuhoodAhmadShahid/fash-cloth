import sys
import os

# Add backend to path
backend_dir = os.path.abspath("backend")
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

print("Testing cv2 import...")
try:
    import cv2
    print("cv2 imported successfully")
except Exception as e:
    print(f"cv2 import failed: {e}")

print("Testing mediapipe import...")
try:
    import mediapipe as mp
    print("mediapipe imported successfully")
except Exception as e:
    print(f"mediapipe import failed: {e}")

print("Testing transformers import...")
try:
    from transformers import pipeline
    print("transformers imported successfully")
except Exception as e:
    print(f"transformers import failed: {e}")
