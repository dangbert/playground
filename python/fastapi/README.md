## FastAPI Playground

This folder defines the backend (python) API for this project.

See https://fastapi.tiangolo.com/tutorial/

### Setup
````bash
pip install poetry
poetry install
````

### Usage

Start dev server, then vist http://localhost:8000
* You can also visit http://localhost:8000/docs (swagger docs)
* And view OpenAPI schema at http://localhost:8000/openapi.json

````bash
# start dev server
uvicorn main:app --reload
````