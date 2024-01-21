#!/usr/bin/env python3
# https://python.langchain.com/docs/expression_language/get_started#rag-search-example
# https://python.langchain.com/docs/expression_language/cookbook/retrieval

import os
import argparse

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, ".env")


def main():
    parser = argparse.ArgumentParser(
        description="Query ChatBot using context queried from provided files."
    )
    parser.add_argument(
        "--query", "-q", type=str, required=True, help="Query question for chat bot."
    )
    parser.add_argument(
        "--input-files",
        "-i",
        action="append",
        help="path to text file(s) to use as context for LLM query",
    )
    args = parser.parse_args()

    load_api_key()
    texts = []
    for path in args.input_files:
        with open(path, "r") as f:
            text = f.read().strip()
            texts.append(text)

    avg_chars = sum([len(t) for t in texts]) / len(texts)

    logger.info(f"loaded {len(texts)} files (averaging {avg_chars:.2f} chars/file)")

    # TODO: support pdfs https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
    # TODO: support folders (using glob)
    logger.info(f"building vectorstore...")
    vectorstore = DocArrayInMemorySearch.from_texts(
        texts,
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    logger.info(f"building chain...")
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo")
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    chain = setup_and_retrieval | prompt | model | output_parser

    logger.info(f"querying chain...")
    # TODO: limit max_tokens by restricting context size
    res = chain.invoke(args.query)
    print(res)


def load_api_key():
    if os.getenv("OPENAI_API_KEY") is None:
        # fallback to reading env file
        from dotenv import load_dotenv

        if not load_dotenv(override=True, dotenv_path=ENV_PATH):
            print(f"failed to load {ENV_PATH}")
            exit(1)
        logger.info(f"loaded {ENV_PATH}")


if __name__ == "__main__":
    main()
