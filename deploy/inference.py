# deploy/inference.py
import os, json
from fastapi import FastAPI
from pydantic import BaseModel

# make repo src importable
from pathlib import Path; import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

# import your core function
from src.genai_flavor_rag import answer

app = FastAPI()

class Query(BaseModel):
    text: str
    top_k: int | None = 12

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/invocations")
def invocations(q: Query):
    return answer(q.text, top_k=q.top_k or 12)
