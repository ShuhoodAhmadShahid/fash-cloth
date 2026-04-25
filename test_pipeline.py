import sys
from transformers import pipeline

try:
    print("Testing pipeline...")
    _gender_model = pipeline(
        "image-classification",
        model="rizvandwiki/gender-classification",
        device=-1,
    )
    print("Pipeline loaded!")
except Exception as e:
    print(f"Pipeline error: {e}")
