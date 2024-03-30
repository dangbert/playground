# danoGPT
>My implementation of Andrej Karpathy's amazing [Tutorial: "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY).  See also his [repo: nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py).

## Setup / Usage

````bash
# download example dataset
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# install dependencies (create virtualenv first if desired)
pip install poetry
poetry install
````

````bash
# train and evaluate bigram model
./bigram.py
# it autodetects the device, but you can manually specify e.g. with:
./bigram.py -d cpu # faster than MPS sadly :(

# you can also try running with accelerate (I didn't see a speedup on macbook at least)
accelerate launch  bigram.py -n 5000
````

## Development Notes:
````bash
# run unit tests
pytest -v

# format code
ruff format .

# fix linting issues
ruff check --fix .

# check type errors
mypy .
````

## See also:
* [notebook tutorial: Transformers and Multi-Head attention](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)
* [paper: Attention is All you Need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
* more advanced tokenizers:
    * [repo: sentencepiece](https://github.com/google/sentencepiece) (encoder from google)
    * [repo: tiktoken](https://github.com/openai/tiktoken) (encoders from OpenAI)