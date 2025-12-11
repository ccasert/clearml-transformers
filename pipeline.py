
from clearml import Task, PipelineController

PROJECT_NAME = "LLM pipeline test"
PIPELINE_NAME = "IMDB Sentiment Analysis Pipeline"

CONTAINER_SETUP_SCRIPT = """
pip install clearml
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
"""
DOCKER_IMAGE = "nersc/pytorch:24.08.01"

# can use Docker overrides for pipeline steps
docker_overrides = {
    'container.image': DOCKER_IMAGE,
    'container.docker_bash_setup_script': CONTAINER_SETUP_SCRIPT
}


pipe = PipelineController(
    name=PIPELINE_NAME,
    project=PROJECT_NAME,
    version="1.2",
)

pipe.set_default_execution_queue('experimental')

pipe.add_parameter(
    name="raw_dataset_id",
    description="The ClearML Dataset ID of the raw IMDB data.",
    default="f38bce9da2994eeb91e37b850a4049ab"
)
pipe.add_parameter(
    name="base_model",
    description="The base Hugging Face model to use for tokenizing and training.",
    default="distilbert-base-uncased"
)

base_preprocess_task = Task.create(
    project_name=PROJECT_NAME, task_name="Base Preprocessing Task",
    script="preprocess.py", docker=DOCKER_IMAGE,
    docker_bash_setup_script=CONTAINER_SETUP_SCRIPT, add_task_init_call=False
)
base_train_task = Task.create(
    project_name=PROJECT_NAME, task_name="Base Training Task",
    script="train.py",
    binary="accelerate launch",
    docker=DOCKER_IMAGE,
    docker_bash_setup_script=CONTAINER_SETUP_SCRIPT,
    add_task_init_call=False
)
base_train_task.set_user_properties(
    num_nodes=2,
    ntasks_per_node=4,
    cpus_per_task=32,
)

base_eval_task = Task.create(
    project_name=PROJECT_NAME, task_name="Base Evaluation Task",
    script="eval.py", docker=DOCKER_IMAGE,
    docker_bash_setup_script=CONTAINER_SETUP_SCRIPT, add_task_init_call=False
)

preprocess_step_name = "step_preprocess"
pipe.add_step(
    name=preprocess_step_name,
    base_task_id=base_preprocess_task.id,
    parameter_override={
        "Args/raw_dataset_id": "${pipeline.raw_dataset_id}",
        "Args/base_model": "${pipeline.base_model}"
    },
    cache_executed_step=True,
)

train_step_name = "step_train"
pipe.add_step(
    name=train_step_name,
    parents=[preprocess_step_name],
    base_task_id=base_train_task.id,
    parameter_override={
        "Args/processed_dataset_id": "${step_preprocess.parameters.General/processed_dataset_id}",
        "Args/base_model": "${pipeline.base_model}"
    },
)


eval_step_name = "step_evaluate"
pipe.add_step(
    name=eval_step_name,
    parents=[train_step_name],
    base_task_id=base_eval_task.id,
    parameter_override={
        "Args/processed_dataset_id": "${step_preprocess.parameters.General/processed_dataset_id}",
        "Args/model_task_id": "${step_train.id}"
    },
)


pipe.start_locally()