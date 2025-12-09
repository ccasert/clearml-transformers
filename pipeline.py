
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

# --- Docker overrides for pipeline steps ---
docker_overrides = {
    'container.image': DOCKER_IMAGE,
    'container.docker_bash_setup_script': CONTAINER_SETUP_SCRIPT
}

# ===================================================================================
# PIPELINE DEFINITION
# ===================================================================================

pipe = PipelineController(
    name=PIPELINE_NAME,
    project=PROJECT_NAME,
    version="1.2",
)

pipe.set_default_execution_queue('experimental')

pipe.add_parameter(
    name="raw_dataset_id",
    description="The ClearML Dataset ID of the raw IMDB data.",
    default="01f89193a4df4df9890dc35fb24e53d5"
)

# --- Define Base Tasks from LOCAL scripts ---
base_preprocess_task = Task.create(
    project_name=PROJECT_NAME, task_name="Base Preprocessing Task",
    script="preprocess.py", docker=DOCKER_IMAGE,
    docker_bash_setup_script=CONTAINER_SETUP_SCRIPT, add_task_init_call=False
)
base_train_task = Task.create(
    project_name=PROJECT_NAME, task_name="Base Training Task",
    script="train.py", docker=DOCKER_IMAGE,
    docker_bash_setup_script=CONTAINER_SETUP_SCRIPT, add_task_init_call=False
)
base_eval_task = Task.create(
    project_name=PROJECT_NAME, task_name="Base Evaluation Task",
    script="eval.py", docker=DOCKER_IMAGE,
    docker_bash_setup_script=CONTAINER_SETUP_SCRIPT, add_task_init_call=False
)

# --- Add steps to the pipeline ---
preprocess_step_name = "step_preprocess"
pipe.add_step(
    name=preprocess_step_name,
    base_task_id=base_preprocess_task.id,
    parameter_override={"Args/raw_dataset_id": "${pipeline.raw_dataset_id}"},
    # task_overrides=docker_overrides,
)

train_step_name = "step_train"
pipe.add_step(
    name=train_step_name,
    parents=[preprocess_step_name],
    base_task_id=base_train_task.id,
    parameter_override={
        "Args/processed_dataset_id": "${{{}.parameters.General/processed_dataset_id}}".format(preprocess_step_name)
    },
    # task_overrides=docker_overrides,
)

eval_step_name = "step_evaluate"
pipe.add_step(
    name=eval_step_name,
    parents=[train_step_name],
    base_task_id=base_eval_task.id,
    parameter_override={
        "Args/processed_dataset_id": "${{{}.parameters.General/processed_dataset_id}}".format(preprocess_step_name),
        "Args/model_task_id": "${{{}.id}}".format(train_step_name)
    },
    # task_overrides=docker_overrides,
)

pipe.start_locally()