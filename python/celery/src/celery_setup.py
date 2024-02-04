#!/usr/bin/env python3

from celery import Celery

capp = Celery(
    "playground",
    # note: /0 means use the database under index 0
    broker="redis://redis:6379/0",
    backend="rpc://",
    include=["proj.tasks"],
)

# Optional configuration, see the application user guide.
# app.conf.update()

if __name__ == "__main__":
    capp.start()
