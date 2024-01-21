# LangChain

This folder is inspired by:
* https://python.langchain.com/docs/expression_language/get_started
* for a slightly different (more detailed) RAG example, see https://python.langchain.com/docs/expression_language/cookbook/retrieval

## Setup

````bash
# copy and edit the secrets file
cp .env.sample .env

# install requirements
pip install -r requirements.txt
````

Note: using and older version of pydantic due to [this issue](https://github.com/langchain-ai/langchain/issues/14585#issuecomment-1855094354).

## RAG Search
> And related references I found on the way...

* https://python.langchain.com/docs/expression_language/get_started#rag-search-example

* [pypi: docarray](https://pypi.org/project/docarray/), [repo](https://github.com/docarray/docarray)
  * A library for representing, storing, searching, sending multimodal data (particularly for AI).

* FAISS (Facebook AI Similarity Search) [site/docs](https://faiss.ai/), [repo](https://ai.meta.com/tools/faiss/)
  * A library to quickly search for embeddings of multimedia documents that are similar to each other.
  * (Homepage lists interesting papers its based off of).

### Ragger
WIP tool for querying ChatGPT using local files as context.