#!/usr/bin/env python3
"""HTTP 问答服务:面向监管制度与统计报表的可信 RAG 问答 API。

端点:
  GET  /          Web 问答界面(src/static 静态页面)
  POST /ask       {"question": "..."} → {answer, evidence[], fidelity, latency_ms}
  GET  /health    存活检查

启动: uvicorn src.api:app --host 127.0.0.1 --port 8000
      浏览器访问 http://127.0.0.1:8000 即可使用 Web 界面
依赖: fastapi uvicorn(pip install fastapi uvicorn)
"""
import json
import os
import sys
import time

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from answer_question import answer, render  # noqa: E402
from llm_client import LLMClient            # noqa: E402
from run_baseline import TableQA, TextRetriever  # noqa: E402

BLOCKS = os.path.join(ROOT, "data", "processed", "text_blocks.jsonl")

app = FastAPI(title="监管制度可信RAG问答", version="1.0")
_state = {}


@app.on_event("startup")
def _load():
    blocks = [json.loads(l) for l in open(BLOCKS, encoding="utf-8")]
    _state["tr"] = TextRetriever(blocks)
    _state["tq"] = TableQA()
    _state["cli"] = LLMClient()
    print("服务就绪:", _state["cli"].info())


class AskIn(BaseModel):
    question: str


@app.post("/ask")
def ask(inp: AskIn):
    t0 = time.time()
    res = answer(inp.question, _state["cli"], _state["tr"], _state["tq"])
    return {**res, "latency_ms": round((time.time() - t0) * 1000)}


@app.get("/health")
def health():
    return {"status": "ok", "provider": _state.get("cli").info() if _state else "loading"}


# 静态 Web 界面挂在路由之后:/ask、/health、/docs 不受影响,GET / 返回 index.html
app.mount("/", StaticFiles(directory=os.path.join(ROOT, "src", "static"), html=True),
          name="static")
