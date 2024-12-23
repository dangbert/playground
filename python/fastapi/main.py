from fastapi import FastAPI, HTTPException
from fastapi import Query
from typing import Annotated

app = FastAPI()


@app.get("/")
async def root():
    return {"hello": "world"}


COURSES = {"example": {"1234567890"}}


@app.get("/join")
# async def join(invite_code: str):
async def join(invite_code: Annotated[str, Query(min_length=10, max_length=10)]):
    for course, codes in COURSES.items():
        if invite_code in codes:
            return {"course": course}
    # 400 error
    raise HTTPException(status_code=400, detail="Invalid invite code")
