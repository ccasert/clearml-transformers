import argparse
import os
import torch
import torch.distributed as dist

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from clearml import Task, Dataset

def setup_ddp():
    """Initializes the DDP process group."""
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Initialize the process group
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)

    print(f"[Process {rank}] World size: {world_size}, Local rank: {local_rank}, Device: {torch.cuda.current_device()}")
    return rank, local_rank, world_size

def train_model(processed_dataset_id: str, base_model: str):
    # 1. Setup the distributed environment first
    rank, local_rank, world_size = setup_ddp()

    # 2. Only the global rank 0 process should manage the ClearML Task
    if rank == 0:
        task = Task.init(project_name="LLM pipeline test", task_name="Model Training")
        logger = task.get_logger()
        logger.report_text(f"Step 2: Training model using processed dataset ID: {processed_dataset_id}")

        hyperparams = {
            'epochs': 3,
            'learning_rate': 1e-5,
            'train_batch_size': 16, # Per-device batch size
            'eval_batch_size': 16,
            'world_size': world_size,
        }
        task.connect(hyperparams)
    else:
        # Other processes wait for rank 0 to finish setup
        dist.barrier()
        hyperparams = {}

    # All processes download the data
    processed_data_path = Dataset.get(dataset_id=processed_dataset_id).get_local_copy()
    tokenized_datasets = load_from_disk(processed_data_path)

    # All processes load the model onto their assigned GPU
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    # NOTE: The Trainer will automatically move the model to the correct device (local_rank)
    # and wrap it in torch.nn.parallel.DistributedDataParallel for you.

    training_args = TrainingArguments(
        output_dir=f"./results_rank_{rank}", # Give each process a unique output dir
        num_train_epochs=hyperparams.get('epochs', 3),
        learning_rate=hyperparams.get('learning_rate', 1e-5),
        per_device_train_batch_size=hyperparams.get('train_batch_size', 16),
        per_device_eval_batch_size=hyperparams.get('eval_batch_size', 16),
        logging_dir=f'./logs_rank_{rank}',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="clearml", # Trainer is smart enough to only report from rank 0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    # All processes participate in training
    trainer.train()

    if rank == 0:
        logger.report_text("Training complete. Best model automatically uploaded as an artifact by ClearML.")

    # Clean up the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dataset_id", type=str, required=True, help="ID of the processed ClearML Dataset")
    parser.add_argument("--base_model", type=str, default="distilbert-base-uncased")
    args = parser.parse_args()

    train_model(args.processed_dataset_id, args.base_model)