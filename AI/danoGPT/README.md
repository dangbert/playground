# danoGPT
>My implementation of Andrej Karpathy's amazing [Tutorial: "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY).

## Setup / Usage

````bash
# download example dataset
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# install dependencies (create virtualenv first if desired)
pip install poetry
poetry install
````

````bash
# test and evaluate v2 model
./v2.py -d cpu


# train and evaluate bigram model
./bigram.py
# it autodetects the device, but you can manually specify e.g. with:
./bigram.py -d cpu # faster than MPS sadly :(

# you can also try running with accelerate (I didn't see a speedup on macbook at least)
accelerate launch bigram.py -n 5000
````

## Development Notes:

````bash
# ensure dev dependencies are installed
poetry install --with dev

# run unit tests, code formatting, linting and type checking
./checks.sh
````

## See also:
* [repo for video lecture](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)
* [repo: nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py) (cleaner more general version of video's repo)
* [notebook tutorial: Transformers and Multi-Head attention](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)
* [paper: Attention is All you Need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
* [Hugging Face: OpenAI GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)
* more advanced tokenizers:
    * [repo: sentencepiece](https://github.com/google/sentencepiece) (encoder from google)
    * [repo: tiktoken](https://github.com/openai/tiktoken) (encoders from OpenAI)