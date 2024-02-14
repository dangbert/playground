#!/usr/bin/env python3
import os
import argparse
import sys
from timeit import default_timer as timer

import utils
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="Modify an image with a text prompt using the SDXL-Turbo model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # parser.add_argument(
    #    "prompt",
    #    type=str,
    #    help="Prompt for image generation",
    #    # default="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
    # )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="path of input image",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--strength",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path of output image",
        default="img_out.jpg",
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
        default=1,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--show",
        action="store_true",
    )
    args = parser.parse_args()
    if args.output:
        assert args.output.endswith(".jpg")

    if args.device is None:
        args.device = utils.get_device()
    torch_dtype = torch.float32 if args.device == "cpu" else torch.float16

    start_time = timer()
    # use from_pipe to avoid consuming additional memory when loading a checkpoint
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch_dtype, variant="fp16"
    )
    pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to(args.device)
    dur_secs = timer() - start_time
    print(f"model loaded in {dur_secs:.3f} seconds")

    # init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
    init_image = load_image(args.input)
    # TODO: try to avoid resizing?
    init_image = init_image.resize((512, 512))

    # prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

    unique = 0
    while True:
        print()
        args.prompt = input("prompt > ").strip()
        try:
            args.strength = float(input("strength > ").strip())
        except ValueError as e:
            print("using previous strength = ", args.strength)
        for i in range(args.num_images):
            start_time = timer()
            image = pipeline(
                args.prompt,
                image=init_image,
                strength=args.strength,
                guidance_scale=0.0,
                num_inference_steps=2,
            ).images[0]
            dur_secs = timer() - start_time
            print(f"\nimage {i+1} generated in {dur_secs:.3f} seconds")

            # make_image_grid([init_image, image], rows=1, cols=2)
            args.output = args.input
            if args.output:
                fname = args.output
                # unique = 0
                while os.path.exists(fname):
                    unique += 1
                    fname = args.output
                    fname = fname[: fname.rfind(".jpg")] + "__" + str(unique) + ".jpg"

                image.save(fname)
                print(f"image wrote to: '{fname}'")

            if args.show:
                image.show()


if __name__ == "__main__":
    main()
