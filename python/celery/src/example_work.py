#!/usr/bin/env python3
import src.tasks as tasks
import random

from celery.result import AsyncResult


def main():
    # https://docs.celeryq.dev/en/stable/getting-started/next-steps.html#calling-tasks
    res = tasks.add.delay(random.randint(1, 100), random.randint(1, 100))
    describe(res)

    res.get(timeout=2)
    describe(res)

    num_list = [random.randint(1, 100000) for _ in range(10)]
    res = tasks.xsum.delay(num_list)
    describe(res)
    res.get(timeout=4)

    describe(res)

    res = tasks.reverse.delay(num_list)
    res.get()  # wait for result
    # celery will actually get the result from the database as the native type it was returned as
    res2 = lookup(res.id)
    print(f"looked up task with result {type(res2.result)}:\n", res2.result)


def lookup(task_id: str):
    """Fetch a result from the backend (sqlite) by task_id."""
    from src.get_celery import app

    res = AsyncResult(task_id, app=app)
    return res


def describe(res: AsyncResult):
    print(
        f"\nid={res.id}: state={res.state}, ready={res.ready()}, successful={res.successful()}, result={res.result}"
    )


if __name__ == "__main__":
    main()
