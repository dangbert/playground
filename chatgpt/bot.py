#!/usr/bin/env python3

import os
import argparse
import openai

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, ".env")

# prices per 1K tokens:
#   https://openai.com/pricing#language-models
PRICES = {
    "gpt-3.5-turbo": 0.002,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="prompt for the chatbot")
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo", help="name of model to use"
    )
    parser.add_argument(
        "-ma",
        "--max-tokens",
        type=int,
        default=500,
        help="max tokens to limit request to",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="open debugger after API request"
    )
    args = parser.parse_args()

    if os.getenv("OPENAI_KEY") is None:
        # fallback to reading env file
        from dotenv import load_dotenv

        if not load_dotenv(override=True, dotenv_path=ENV_PATH):
            print(f"failed to load {ENV_PATH}")
            exit(1)
    openai.api_key = os.getenv("OPENAI_KEY")

    completion = openai.ChatCompletion.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
    )

    if len(completion.choices) > 1:
        print(
            f"NOTE: {completion.choices} responses returned, printing just the first\n"
        )

    res = completion.choices[0].message.content
    print(res)

    print(f"\n===================")
    usage = completion.usage
    if args.model in PRICES:
        price = usage["total_tokens"] / 1000 * PRICES[args.model]
        print(f"price: ${price:.5f}")
    else:
        print(f"price unknown for model '{args.model}'")
    # print(f"usage = {usage}")
    print(
        f"total tokens: {usage['total_tokens']} ({usage['prompt_tokens']} prompt, {usage['completion_tokens']} response)"
    )

    if args.debug:
        import pdb

        pdb.set_trace()
    print()


if __name__ == "__main__":
    main()
