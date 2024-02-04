#!/usr/bin/env python3
from celery import Celery

REDIS_PORT = 6380
REDIS_HOST = "localhost"

app = Celery(
    "playground",
    # note: /0 means use the database under index 0
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend="db+sqlite:///db.sqlite3",
    include=["src.tasks"],
)

# Optional configuration, see the application user guide.
# app.conf.update()

if __name__ == "__main__":
    # TODO: circular import, we may not even need to run this script directly
    app.start()
