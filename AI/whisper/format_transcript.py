#!/usr/bin/env python3
# print a transcript prediction.json to stdout with timestamps shown about every 3 minutes

import os
import argparse
import sys
import json
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    last_print_timestamp = 0
    # Read the content of prediction.json
    with open("prediction.json", "r") as file:
        data = json.load(file)

    # Loop through the transcript data and print the combined transcript
    for segment in data:
        # Extract the start time of the segment
        start_time = segment["timestamp"][0]

        # Print the transcript text
        print(segment["text"], end=" ")

        # print timestamp about every 3 min
        if start_time >= last_print_timestamp + 3 * 60:
            current_time = time.strftime("%H:%M:%S", time.gmtime(start_time))
            print(f"\n\n[{current_time}]")
            last_print_timestamp = start_time - (start_time % 60)

    # Print the final current time after the transcript completion


if __name__ == "__main__":
    main()
