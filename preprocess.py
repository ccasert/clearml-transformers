import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from clearml import Task, Dataset
import re

def preprocess_data(raw_dataset_id: str, base_model: str):
    task = Task.current_task()
    logger = task.get_logger()

    logger.report_text(f"Step 1: Preprocessing data from raw dataset ID: {raw_dataset_id}")
    logger.report_text(f"Using base model for tokenizer: {base_model}")

    # Get a local copy of the raw data
    raw_data_path = Dataset.get(dataset_id=raw_dataset_id).get_local_copy()

    data_files = {
        "train": os.path.join(raw_data_path, "train.csv"),
        "test": os.path.join(raw_data_path, "test.csv"),
    }
    hf_dataset = load_dataset("csv", data_files=data_files)
    logger.report_text(f"Loaded raw dataset. Train examples: {len(hf_dataset['train'])}, Test examples: {len(hf_dataset['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)

    # Clean the base_model string to be a valid dataset name component
    cleaned_model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_model.split("/")[-1])
    dataset_name = f"processed_{cleaned_model_name}_padded" # Added suffix for clarity

    processed_dataset = Dataset.create(
        dataset_project=task.get_project_name(),
        dataset_name=dataset_name,
        parent_datasets=[raw_dataset_id]
    )

    processed_folder = "processed_data"
    tokenized_datasets.save_to_disk(processed_folder)
    processed_dataset.add_files(processed_folder)

    logger.report_text("Uploading processed dataset to ClearML...")
    processed_dataset.upload()
    processed_dataset.finalize()

    logger.report_text(f"Created processed dataset '{dataset_name}' with ID: {processed_dataset.id}")
    return processed_dataset.id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset_id", type=str, required=True, help="ID of the raw ClearML Dataset")
    parser.add_argument("--base_model", type=str, default="distilbert-base-uncased", help="Base model for tokenizer")
    args = parser.parse_args()

    Task.init(project_name="LLM pipeline test", task_name="Data Preprocessing")

    processed_id = preprocess_data(args.raw_dataset_id, args.base_model)

    Task.current_task().set_parameter("processed_dataset_id", processed_id)