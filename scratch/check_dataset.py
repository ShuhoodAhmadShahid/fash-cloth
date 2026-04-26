import kagglehub
from kagglehub import KaggleDatasetAdapter

print("Loading dataset metadata...")
try:
    df = kagglehub.load_dataset(
      KaggleDatasetAdapter.PANDAS,
      "agrigorev/clothing-dataset-full",
      "images.csv",
    )
    print("Columns:", df.columns.tolist())
    print("First 5 labels:", df['label'].unique()[:10].tolist())
except Exception as e:
    print("Error:", e)
