> A script for quickly generating images with sdxl-turbo.

* https://huggingface.co/stabilityai/sdxl-turbo
* [paper](https://stability.ai/research/adversarial-diffusion-distillation)

````bash
# setup
pip install diffusers transformers accelerate --upgrade

# usage
./text_to_image.py "A skyscraper made of cheese" # --show
# NOTE: write to out.jpg by default
````

