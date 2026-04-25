import os
from huggingface_hub import hf_hub_download, snapshot_download
import pandas as pd

def download_fashion_data():
    """
    Downloads a fashion dataset from Hugging Face for real product recommendations.
    Using 'marmat/fashion-dataset' which is a popular one for this purpose.
    """
    print("Initiating fashion dataset download from Hugging Face...")
    
    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), "..", "backend", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # 1. Download styles.csv (metadata)
        # Note: You need a HF token for some datasets
        styles_path = hf_hub_download(
            repo_id="marmat/fashion-dataset", 
            filename="styles.csv",
            repo_type="dataset",
            local_dir=data_dir
        )
        print(f"Fashion metadata saved to: {styles_path}")
        
        # 2. To download images (optional, as they are many):
        # snapshot_download(repo_id="marmat/fashion-dataset", repo_type="dataset", allow_patterns="images/*.jpg")
        
        print("\nDataset successfully initialized!")
        print("You can now index these products in your FAISS index for real-world recommendations.")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Tip: Make sure you have 'huggingface_hub' and 'pandas' installed.")

if __name__ == "__main__":
    download_fashion_data()
