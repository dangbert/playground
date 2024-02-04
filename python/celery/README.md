# Celery

Simple demo of using Celery to run tasks and access their result.  In this example Celery is configured to use Redis for managing tasks queues (a.k.a as a "broker"), and sqlite as the (optional) "backend" for storing results (the tables "celery_taskmeta" and "celery_tasksetmeta" are created/used by default).

References:
* https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html9
* https://docs.celeryq.dev/en/stable/getting-started/next-steps.html
* https://docs.celeryq.dev/en/stable/userguide/index.html

## Setup / Usage
````bash
# start redis
docker compose up -d 

pip install poetry
poetry install
# enter virtual environment
poetry shell

# run celery (looks at tasks.py:app object)
#   https://docs.celeryq.dev/en/stable/getting-started/next-steps.html#about-the-app-argument
cd src
celery -A tasks worker --loglevel=INFO

# now leaving our background worker running
# enter a new terminal and start some example tasks:
poetry shell
./example_work.py

# see running nodes:
celery -A tasks inspect active

# view redis if desired
redis-cli -p 6380
keys *
# outputs:
#1) "_kombu.binding.celery.pidbox"
#2) "_kombu.binding.celery"
#3) "_kombu.binding.celeryev"
````