# Celery

References:
* https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html9

## Setup / Usage
````bash
# start redis
docker compose up -d 

pip install poetry
poetry install
# enter virtual environment
poetry shell

# run celery (looks at tasks.py:app object)
celery -A tasks worker --loglevel=INFO

# view redis if desired
redis-cli -p 6380
````