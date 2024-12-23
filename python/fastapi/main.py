from fastapi import FastAPI, HTTPException
from fastapi import Query
from typing import Annotated
from pydantic import BaseModel

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


class Item(BaseModel):
    x: str | float | int


@app.post("/test")
async def test(item: Item):
    return {"x": str(type(item.x))}


# note these always were strings:
# async def test(x: Union[int, float, str]):
# async def test(x: int | float | str):
