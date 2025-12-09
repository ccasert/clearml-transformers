import argparse
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from clearml import Task, Dataset

def train_model(processed_dataset_id: str, base_model: str):
    task = Task.current_task()
    print(f"Step 2: Training model using processed dataset ID: {processed_dataset_id}")

    # Connect hyperparameters for reproducibility
    hyperparams = {
        'epochs': 3,
        'learning_rate': 1e-5,
        'train_batch_size': 16,
        'eval_batch_size': 16,
    }
    task.connect(hyperparams)

    # Get the processed data from the previous step
    processed_data_path = Dataset.get(dataset_id=processed_dataset_id).get_local_copy()
    tokenized_datasets = load_from_disk(processed_data_path)

    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    # Define training arguments. This is the key part for artifact logging.
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=hyperparams['epochs'],
        learning_rate=hyperparams['learning_rate'],
        per_device_train_batch_size=hyperparams['train_batch_size'],
        per_device_eval_batch_size=hyperparams['eval_batch_size'],
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="clearml",
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    # Start the training
    print("Starting model training...")
    trainer.train()
    print("Training complete. Best model automatically uploaded as an artifact by ClearML.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dataset_id", type=str, required=True, help="ID of the processed ClearML Dataset")
    parser.add_argument("--base_model", type=str, default="distilbert-base-uncased")
    args = parser.parse_args()

    # Initialize the ClearML Task for this step
    Task.init(project_name="LLM pipeline test", task_name="Model Training")

    train_model(args.processed_dataset_id, args.base_model)