#!/usr/bin/env python3
from greenlet import greenlet


def task1():
    print("Task 1: Step 1")
    g2.switch()
    print("Task 1: Step 2")
    g2.switch()


def task2():
    print("Task 2: Step 1")
    g1.switch()
    print("Task 2: Step 2")


"""
simple example printing:
Task 1: Step 1
Task 2: Step 1
Task 1: Step 2
Task 2: Step 2
"""
# greenlet objects are created for each task but not ran
g1 = greenlet(task1)
g2 = greenlet(task2)
# now start g1 (which will itself start g2)
g1.switch()
