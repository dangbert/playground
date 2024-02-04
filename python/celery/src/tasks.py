from src.get_celery import app
import time


@app.task
def add(x, y):
    # simulate slow task
    time.sleep(0.01)
    return x + y


@app.task
def xsum(numbers):
    time.sleep(0.5)
    return sum(numbers)


@app.task
def reverse(numbers):
    time.sleep(0.5)
    return numbers[::-1]
