#!/usr/bin/env python3
import src.tasks as tasks
import random


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
    breakpoint()


def describe(res):
    print(f"\n{res.id}: {type(res)} state={res.state} result={res.result}")
    print(f"sucessful result: {res.successful()}")


if __name__ == "__main__":
    main()
