#!/usr/bin/env python3
"""
learn/experiment with multiprocessing in python
https://www.youtube.com/watch?v=YOhrIov7PZA

multiprocessing is helpful for CPU bound code.
"""
from concurrent.futures import process
from multiprocessing import Process, cpu_count, Array
import time
from typing import List

def main():
    print(f'note: this computer has {cpu_count()} cpus')
    # count to 1 billion
    #COUNT_TOTAL = pow(10, 9)
    COUNT_TOTAL = pow(10, 8)
    start = time.perf_counter()
    NUM_PROCESSES = 4

    COUNT_EACH = int(COUNT_TOTAL / NUM_PROCESSES)
    print(f'creating {NUM_PROCESSES} processes, each will count to {COUNT_EACH}')
    plist: List[Process] = []

    # use a "shared variable" to let each thread store data
    #   https://docs.python.org/3/library/multiprocessing.html#sharing-state-between-processes
    arr = Array('d', range(NUM_PROCESSES))
    for i in range(NUM_PROCESSES):
        plist.append(Process(target=counter, args=(COUNT_EACH, arr, i)))
        plist[-1].start()


    # have main process wait for processes to finish
    print('\nwaiting on processes...')
    for p in plist:
        p.join()
    stop = time.perf_counter()
    print(f'\nall done (in {(stop - start):.4f} secs)')
    print("cpu_times of each process = ")
    print(arr[:])

def counter(num: int, arr: Array, arr_index: int):
    cpu_time = time.perf_counter()
    count = 0
    while count < num:
        count += 1
    cpu_time = time.perf_counter() - cpu_time

    # store result
    arr[arr_index] = cpu_time


if __name__ == "__main__":
    # prevent child process from executing our module (e.g. in windows)
    main()