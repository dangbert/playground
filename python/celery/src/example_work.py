#!/usr/bin/env python3
import src.tasks as tasks
import random

from celery.result import AsyncResult


def main():
    # https://docs.celeryq.dev/en/stable/getting-started/next-steps.html#calling-tasks
    res = tasks.add.delay(random.randint(1, 100), random.randint(1, 100))
    describe(res)

    res.get(timeout=10)
    describe(res)

    num_list = [random.randint(1, 100000) for _ in range(10)]
    res = tasks.xsum.delay(num_list)
    describe(res)
    res.get(timeout=30)

    describe(res)

    res = tasks.reverse.delay(num_list)
    res.get()  # wait for result
    # celery will actually get the result from the database as the native type it was returned as
    res2 = lookup(res.id)
    print(f"looked up task with result {type(res2.result)}:\n", res2.result)

    # raw = lookup_raw(res.id)
    # print(raw)


def lookup(task_id: str):
    """Fetch a result from the backend (sqlite) by task_id."""
    from src.get_celery import app

    res = AsyncResult(task_id, app=app)
    return res


def lookup_raw(task_id: str):
    # here we do our own query as an example of using celery's ORM models directly
    from celery.backends.database import models as celery_models
    from sqlalchemy.orm import Session
    from src.get_celery import app

    # create SQLAlchemy session against backend URI
    # TODO: this doesn't work:
    # see instead http://www.prschmid.com/2013/04/using-sqlalchemy-with-celery-tasks.html
    session = Session(bind=app.backend.engine)

    # Use the session for database operations with Celery's models
    result = (
        session.query(celery_models.TaskResult)
        .filter_by(task_id="your_task_id")
        .first()
    )
    return result


def describe(res: AsyncResult):
    print(
        f"\nid={res.id}: state={res.state}, ready={res.ready()}, successful={res.successful()}, result={res.result}"
    )


if __name__ == "__main__":
    main()
