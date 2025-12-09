from datasets import load_dataset
from clearml import Dataset
import os

project_name = "LLM pipeline test"
dataset_name = "imdb_raw"

output_folder = "raw_imdb_data"
os.makedirs(output_folder, exist_ok=True)
train_path = os.path.join(output_folder, "train.csv")
test_path = os.path.join(output_folder, "test.csv")

raw_dataset = Dataset.create(
    dataset_project=project_name,
    dataset_name=dataset_name
)

imdb_dataset = load_dataset("imdb")
imdb_dataset["train"].to_csv(train_path, index=False)
imdb_dataset["test"].to_csv(test_path, index=False)

raw_dataset.add_files(output_folder)

raw_dataset.upload()
raw_dataset.finalize()

print(f"Done. Dataset ID: {raw_dataset.id}")
