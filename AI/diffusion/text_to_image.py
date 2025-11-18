#!/usr/bin/env python3
import os
import argparse
import sys
from timeit import default_timer as timer

import utils
from diffusers import AutoPipelineForText2Image
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_NAME = "stabilityai/sdxl-turbo"

#MODEL_NAME = "Qwen/Qwen-Image"
#MODEL_NAME = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"

def main():
    parser = argparse.ArgumentParser(
        description="Generate an image from a text prompt using the SDXL-Turbo model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # parser.add_argument(
    #    "prompt",
    #    type=str,
    #    help="Prompt for image generation",
    #    # default="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
    # )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="output directory or filename",
        default="./text_out.jpg",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,  # "cpu" | "cuda" | "mps"
        help="Manually specify device to run on (auto detects otherwise).",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        help="name of model to run",
        default=MODEL_NAME,
    )
    parser.add_argument(
        "-n",
        "--num_images",
        type=int,
        default=1,
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
        args.model_name, torch_dtype=torch_dtype, variant="fp16"
    )

    pipe.to(args.device)
    dur_secs = timer() - start_time
    print(f"model loaded in {dur_secs:.3f} seconds: '{args.model_name}'")

    unique = 0
    while True:
        print()
        args.prompt = input("prompt > ").strip()
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
                while os.path.exists(fname):
                    unique += 1
                    fname = args.output
                    fname = fname[: fname.rfind(".jpg")] + str(unique) + ".jpg"
                image.save(fname)
                print(f"image wrote to: '{fname}'")

            if args.show:
                image.show()


if __name__ == "__main__":
    main()
