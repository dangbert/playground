from src.celery import app


@app.task
def add(x, y):
    return x + y


@app.task
def xsum(numbers):
    return sum(numbers)
