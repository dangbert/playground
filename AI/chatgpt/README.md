# Experiment with chatgpt API
* [based on this article](https://medium.com/geekculture/a-simple-guide-to-chatgpt-api-with-python-c147985ae28)

## Setup
[First create your OpenAI API key here](https://beta.openai.com/account/api-keys)

````bash
# install dependencies
pip install -r requirements.txt

cp .env.sample .env
# now edit this file to contain your personal API key
````

## Usage

````bash
./bot.py -p "example prompt"

# alternatively you can set API key manually instead of using .env file:
OPENAI_KEY=CHANGE_ME ./bot.py "hello"
````