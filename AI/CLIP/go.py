#!/usr/bin/env python3
# https://github.com/openai/CLIP?tab=readme-ov-file#usage

import os
import argparse
import sys

import torch
import clip
from PIL import Image
import time
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_EXTS = ["jpg", "jpeg", "png"]
MODEL_NAME = "RN50"

CACHE_PATH = os.path.join(SCRIPT_DIR, "all_features.pt")

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="path of photos folder",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        help="target image to search for neighbors of",
    )
    print(f"available models:")
    print(clip.available_models())
    args = parser.parse_args()

    assert os.path.isdir(args.input_dir)
    img_paths = find_images(args.input_dir)
    assert os.path.isfile(args.target)
    args.target = os.path.abspath(args.target)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL_NAME, device=device)
    #img_paths = img_paths[:20] # for now
    #target_path = img_paths[0]
    with torch.no_grad():
        dur = 0
        all_features: dict[str, torch.Tensor] = dict()

        def encode_image(fname: str) -> torch.Tensor:
            image = preprocess(Image.open(fname)).unsqueeze(0).to(device)
            features = model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
            return features


        # target_features = encode_image(args.target)
        for i, img_path in enumerate(img_paths):
            start_time = time.perf_counter()
            img_features = encode_image(img_path)
            dur += time.perf_counter() - start_time

            all_features[os.path.basename(img_path)] = img_features
            if i % 5 == 0:
                print(f"at image {i}/{len(img_paths)}, averaging {(dur/len(all_features)):.4f} sec/image")
    
        torch.save(all_features, CACHE_PATH)
        print(f"wrote '{CACHE_PATH}'")
        x = torch.stack(list(all_features.values())).to(device)

        def query_similar(fname: str) -> str:
            """Given an image, return the name of the most similar image."""
            if fname in all_features:
                target_features = all_features[fname]
            else:
                target_features = encode_image(fname)

            sims =  (x @ target_features.T).squeeze(1)  # Compute cosine similarity
            print(f"{sims.shape=}")
            top5_values, top5_indices = sims.topk(5, largest=True, axis=0)

            k_idx_map = dict()
            i = 0
            for name in all_features.keys():
                k_idx_map[i] = name
                i += 1

            print(f"\n\nmost similar images to {fname}:")
            # get matching filenames
            matches = [k_idx_map[i] for i in top5_indices.flatten().tolist()]
            print("\n".join(matches))
        
        query_similar(args.target)
        while True:
            target = input(f"image path > ").strip()
            if not os.path.isabs(target):
                target = os.path.join(args.input_dir, target)

            try:
                query_similar(target)
            except FileNotFoundError:
                print(f"file not found")


def find_images(input_dir: str) -> list[str]:
    img_paths: list[str] = []
    for filename in os.listdir(input_dir):
        fpath = os.path.join(input_dir, filename)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fpath)[1].lstrip(".")
        if ext.lower() not in SUPPORTED_EXTS:
            continue
        img_paths.append(os.path.abspath(fpath))
    
    img_paths = sorted(img_paths)
    print(f"found {len(img_paths)} images")
    print(f"'{img_paths[0]}' -> '{img_paths[-1]}'")
    return img_paths



if __name__ == "__main__":
    main()
