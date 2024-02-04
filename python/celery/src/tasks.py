from .celery_setup import capp

@capp.task
def add(x, y):
    return x + y


@capp.task
def xsum(numbers):
    return sum(numbers)
