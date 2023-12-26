#!/usr/bin/env python3
import os
import argparse
import sys
from timeit import default_timer as timer

from diffusers import AutoPipelineForText2Image
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        type=str,
        help="Prompt for image generation",
        # default="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path of output image",
        default="out.jpg",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",  # "cuda" | "mps"
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
    )
    args = parser.parse_args()

    start_time = timer()

    print(f"using device: '{args.device}'")
    if args.device != "cpu":
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        )
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float32, variant="fp16"
        )

    pipe.to(args.device)
    dur_secs = timer() - start_time
    print(f"model loaded in {dur_secs:.3f} seconds")

    start_time = timer()
    image = pipe(prompt=args.prompt, num_inference_steps=1, guidance_scale=0.0).images[
        0
    ]
    dur_secs = timer() - start_time
    print(f"\nimage generated in {dur_secs:.3f} seconds")

    if args.output:
        image.save(args.output)
        print(f"image wrote to: '{args.output}'")

    if args.show:
        image.show()


if __name__ == "__main__":
    main()
