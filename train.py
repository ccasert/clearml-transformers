import argparse
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from clearml import Task, Dataset
from accelerate import Accelerator

def train_model(processed_dataset_id: str, base_model: str):
    # Initialize Accelerator to manage the multi-GPU environment
    accelerator = Accelerator()

    # Use the main process for all setup and logging tasks.
    if accelerator.is_main_process:
        task = Task.init(project_name="LLM pipeline test", task_name="Model Training")
        logger = task.get_logger()
        logger.report_text(f"Step 2: Training model using processed dataset ID: {processed_dataset_id}")

        hyperparams = {
            'epochs': 3,
            'learning_rate': 1e-5,
            'train_batch_size': 16,
            'eval_batch_size': 16,
        }
        task.connect(hyperparams)
    else:
        hyperparams = {}

    processed_data_path = Dataset.get(dataset_id=processed_dataset_id).get_local_copy()
    tokenized_datasets = load_from_disk(processed_data_path)

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=hyperparams.get('epochs', 3),
        learning_rate=hyperparams.get('learning_rate', 1e-5),
        per_device_train_batch_size=hyperparams.get('train_batch_size', 16),
        per_device_eval_batch_size=hyperparams.get('eval_batch_size', 16),
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="clearml",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    accelerator.print(f"Process {accelerator.process_index} starting model training...")
    trainer.train()

    if accelerator.is_main_process:
        logger.report_text("Training complete. Best model automatically uploaded as an artifact by ClearML.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dataset_id", type=str, required=True, help="ID of the processed ClearML Dataset")
    parser.add_argument("--base_model", type=str, default="distilbert-base-uncased")
    args = parser.parse_args()

    train_model(args.processed_dataset_id, args.base_model)