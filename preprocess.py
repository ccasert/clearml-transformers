import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from clearml import Task, Dataset

def preprocess_data(raw_dataset_id: str, base_model: str):
    task = Task.current_task()

    print(f"Step 1: Preprocessing data from raw dataset ID: {raw_dataset_id}")

    # Get a local copy of the raw data
    raw_data_path = Dataset.get(dataset_id=raw_dataset_id).get_local_copy()

    data_files = {
        "train": os.path.join(raw_data_path, "train.csv"),
        "test": os.path.join(raw_data_path, "test.csv"),
    }
    hf_dataset = load_dataset("csv", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)

    processed_dataset = Dataset.create(
        dataset_project=task.get_project_name(),
        dataset_name=f"{task.get_project_name()}_processed",
        parent_datasets=[raw_dataset_id]
    )

    processed_folder = "processed_data"
    tokenized_datasets.save_to_disk(processed_folder)
    processed_dataset.add_files(processed_folder)

    processed_dataset.upload()
    processed_dataset.finalize()

    print(f"Created processed dataset with ID: {processed_dataset.id}")
    return processed_dataset.id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset_id", type=str, required=True, help="ID of the raw ClearML Dataset")
    parser.add_argument("--base_model", type=str, default="distilbert-base-uncased", help="Base model for tokenizer")
    args = parser.parse_args()

    # This task will be executed remotely by the pipeline
    Task.init(project_name="LLM pipeline test", task_name="Data Preprocessing")

    processed_id = preprocess_data(args.raw_dataset_id, args.base_model)

    Task.current_task().set_parameter("processed_dataset_id", processed_id)