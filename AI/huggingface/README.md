## Setup

Setup environment with dependencies:

````bash
python3 -m pip install virtualenv
python3 -m virtualenv .venv
. .venv/bin/activate

pip install poetry
poetry config virtualenvs.in-project true
poetry install
````

## Usage:

Start by reading/running [./nlp_course.ipynb](./nlp_course.ipynb)

````bash
# optionally configure wandb project as desired
export WANDB_ENTITY="your-entity"
export WANDB_PROJECT="your-project-name"

./finetune.py # --wandb
````
