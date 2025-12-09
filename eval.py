import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from clearml import Task, Dataset, Model

def evaluate_model(model_task_id: str, processed_dataset_id: str):
    """
    Downloads a trained model and a dataset, evaluates the model,
    and uploads performance metrics and plots as artifacts.
    """
    task = Task.current_task()
    logger = task.get_logger()

    print(f"Step 3: Evaluating model from task ID: {model_task_id}")

    # --- Download Inputs ---
    model_path = Model(task_id=model_task_id).get_local_copy()
    processed_data_path = Dataset.get(dataset_id=processed_dataset_id).get_local_copy()

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenized_datasets = load_from_disk(processed_data_path)
    test_dataset = tokenized_datasets["test"]

    # --- Run Predictions ---
    eval_trainer = Trainer(model=model, args=TrainingArguments(output_dir="./eval_results"))
    predictions = eval_trainer.predict(test_dataset)

    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    # --- Calculate and Log Metrics ---
    metrics_dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": precision_recall_fscore_support(y_true, y_pred, average='binary')[2],
        "precision": precision_recall_fscore_support(y_true, y_pred, average='binary')[0],
        "recall": precision_recall_fscore_support(y_true, y_pred, average='binary')[1],
    }

    print(f"Evaluation Metrics: {metrics_dict}")
    logger.report_scalar_dict(
        title="Performance Metrics",
        series="evaluation",
        iteration=1,
        values=metrics_dict
    )

    # --- Create and Upload Artifacts ---
    # 1. Confusion Matrix Plot (using only Matplotlib)
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Negative', 'Positive']
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    logger.report_matplotlib_figure(
        title="Analysis",
        series="Confusion Matrix",
        figure=fig,
        iteration=1
    )
    print("Confusion matrix plot uploaded as an artifact.")

    # 2. JSON Metrics File
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    task.upload_artifact(name='evaluation_summary', artifact_object='evaluation_metrics.json')
    print("Metrics JSON file uploaded as an artifact.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_task_id", type=str, required=True, help="Task ID of the trained model")
    parser.add_argument("--processed_dataset_id", type=str, required=True, help="ID of the processed ClearML Dataset")
    args = parser.parse_args()

    Task.init(project_name="LLM pipeline test", task_name="Model Evaluation")

    evaluate_model(args.model_task_id, args.processed_dataset_id)