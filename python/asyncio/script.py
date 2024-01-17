#!/usr/bin/env python3

import asyncio

async def main():
    task1 = asyncio.create_task(func1())
    task2 = asyncio.create_task(func2())

    print("awaiting tasks...\n", flush=True)
    await task1
    count = await task2
    print(f"\nmain: received count {count} from task2")

async def func1():
    print("func1: sleeping", flush=True)
    await asyncio.sleep(5)
    print("func1: done!", flush=True)

async def func2():
    print("func2: counting", flush=True)
    count = 0
    for _ in range(1000000):
        count += 1

    print(f"func2: done (counted to {count})", flush=True)
    return count



if __name__ == "__main__":
    asyncio.run(main())
