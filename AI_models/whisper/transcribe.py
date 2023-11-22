#!/usr/bin/env python3

import os
import argparse
import sys
import whisper
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fname",
        type=str,
        help="filename to transcribe",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model version to use",
        default="tiny.en"
    )
    args = parser.parse_args()

    print(f"using model {args.model}")
    model = whisper.load_model(args.model)
    print(f"transcribing '{args.fname}'")
    dur = time.time()
    res = model.transcribe(args.fname)
    dur = time.time() - dur

    print(f"\ntranscription (took {dur:.2f} seconds):")
    print(res['text'])


if __name__ == "__main__":
    main()
