#!/usr/bin/env python3
"""
learn/experiment with multithreading in python
https://www.youtube.com/watch?v=3dEPY3HiPtI

multithreading is helpful for I/O bound code, but not CPU bound code!
"""

import threading
import time


def paint(name):
    time.sleep(2)
    print(f"{name} wall painted")


# this will take about 2 secs total:
print(f"initial thread count: {threading.active_count()} (just the main thread)")  # 1

start = time.perf_counter()
t1 = threading.Thread(target=paint, args=("left",))
t2 = threading.Thread(target=paint, args=("right",))
t3 = threading.Thread(target=paint, args=("ceiling",))
t1.start()
t2.start()
t3.start()

print(f"new thread count: {threading.active_count()}, thread list:")  # 4
print(threading.enumerate())

# optionally we can wait for all threads to complete before running code below
print("\nwaiting on threads...")
for t in [t1, t2, t3]:
    t.join()

stop = time.perf_counter()
print(f"\nall done (in {(stop - start):.4f} secs)")
