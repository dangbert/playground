#!/usr/bin/env python3
import os
import argparse
import sys
from timeit import default_timer as timer

import utils
from diffusers import AutoPipelineForText2Image
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="Generate an image from a text prompt using the SDXL-Turbo model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        default=None,  # "cpu" | "cuda" | "mps"
        help="Manually specify device to run on (auto detects otherwise).",
    )
    parser.add_argument(
        "-n",
        "--num_images",
        type=int,
        default=1,  # "cpu" | "cuda" | "mps"
        help="Number of images to generate",
    )
    parser.add_argument(
        "--show",
        action="store_true",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=1,
        help="Number of inference steps to run, increase for better quality but slower generation",
    )
    args = parser.parse_args()

    if args.output:
        assert args.output.endswith(".jpg")

    start_time = timer()
    if args.device is None:
        args.device = utils.get_device()

    print(f"using device: '{args.device}'")
    torch_dtype = torch.float32 if args.device == "cpu" else torch.float16
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch_dtype, variant="fp16"
    )

    pipe.to(args.device)
    dur_secs = timer() - start_time
    print(f"model loaded in {dur_secs:.3f} seconds")

    for i in range(args.num_images):
        start_time = timer()
        res = pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            # default is 512x512 but seems good at 1920x1080 resolution (at least with 3 inference steps)
            height=args.height,
            width=args.width,
        )
        image = res.images[0]
        dur_secs = timer() - start_time
        print(f"\nimage {i+1} generated in {dur_secs:.3f} seconds")

        if args.output:
            fname = args.output
            if args.num_images > 1:
                fname = fname[: fname.rfind(".jpg")] + str(i + 1) + ".jpg"
            image.save(fname)
            print(f"image wrote to: '{fname}'")

        if args.show:
            image.show()


if __name__ == "__main__":
    main()
