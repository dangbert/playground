#!/usr/bin/env python3
"""
learn/experiment with multiprocessing in python
https://www.youtube.com/watch?v=YOhrIov7PZA

multiprocessing is helpful for CPU bound code.
"""
from concurrent.futures import process
from multiprocessing import Process, cpu_count
import time

def main():
    print(f'note: this computer has {cpu_count()} cpus')
    # count to 1 billion
    COUNT_TOTAL = pow(10, 9)
    start = time.perf_counter()
    NUM_PROCESSES = 4

    COUNT_EACH = int(COUNT_TOTAL / NUM_PROCESSES)
    print(f'creating {NUM_PROCESSES} processes, each will count to {COUNT_EACH}')
    plist = []
    for _ in range(NUM_PROCESSES):
        plist.append(Process(target=counter, args=(COUNT_EACH,)))
        plist[-1].start()


    # have main process wait for processes to finish
    print('\nwaiting on processes...')
    for p in plist:
        p.join()
    stop = time.perf_counter()
    print(f'\nall done (in {(stop - start):.4f} secs)')

def counter(num):
    count = 0
    while count < num:
        count += 1

if __name__ == "__main__":
    # prevent child process from executing our module (e.g. in windows)
    main()