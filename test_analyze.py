import sys
import os
import numpy as np
from PIL import Image

backend_dir = os.path.abspath("backend")
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from services.image_processing import analyze_image

# Create dummy image
img = Image.new('RGB', (512, 512), color = (73, 109, 137))

try:
    print("Testing analyze_image...")
    res = analyze_image(img)
    print("Result:", res)
except Exception as e:
    print(f"analyze_image exception: {e}")
