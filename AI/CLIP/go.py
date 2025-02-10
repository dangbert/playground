#!/usr/bin/env python3
# https://github.com/openai/CLIP?tab=readme-ov-file#usage

import os
import argparse
import sys

import torch
import clip
from PIL import Image
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_EXTS = ["jpg", "jpeg", "png"]
MODEL_NAME = "RN50"

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="path of photos folder",
    )
    print(f"available models:")
    print(clip.available_models())
    args = parser.parse_args()

    assert os.path.isdir(args.input_dir)
    img_paths = find_images(args.input_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL_NAME, device=device)
    with torch.no_grad():
        # target_path = img_paths[0]
        dur = 0
        all_features: list[torch.Tensor] = []
        for i, img_path in enumerate(img_paths):
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            start_time = time.perf_counter()
            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            dur += time.perf_counter() - start_time

            all_features.append(img_features)
            if i % 5 == 0:
                print(f"at image {i}/{len(img_paths)}, averaging {(dur/len(all_features)):.4f} sec/image")


def find_images(input_dir: str) -> list[str]:
    img_paths: list[str] = []
    for filename in os.listdir(input_dir):
        fpath = os.path.join(input_dir, filename)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fpath)[1].lstrip(".")
        if ext.lower() not in SUPPORTED_EXTS:
            continue
        img_paths.append(fpath)
    
    img_paths = sorted(img_paths)
    print(f"collected {len(img_paths)} images")
    print(f"'{img_paths[0]}' -> '{img_paths[-1]}'")
    return img_paths



if __name__ == "__main__":
    main()
