import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "agrigorev/clothing-dataset-full",
  "images.csv",
)
print(df['label'].unique().tolist())
