#!/usr/bin/env python3
"""
qwen_server.py — OpenAI-compatible server using llama-cpp-python as library.
Avoids llama_cpp.server DNS resolution bug on systems where hostname
doesn't resolve. Serves on 127.0.0.1:8001.

Endpoints:
  GET  /health                  — liveness check
  GET  /v1/models               — list model
  POST /v1/completions          — text completion (used by qwen_client.py)
  POST /v1/chat/completions     — chat completion
"""

import argparse
import time
import uuid
import json
import asyncio
import logging
from typing import List, Optional
from threading import Lock

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--n_gpu_layers", type=int, default=35)
parser.add_argument("--n_ctx", type=int, default=8192)
parser.add_argument("--model_alias", default="qwen2.5-14b")
args = parser.parse_args()

# ── Load model ────────────────────────────────────────────────────────────────
logger.info(f"Loading model from {args.model} ...")
llm = Llama(
    model_path=args.model,
    n_gpu_layers=args.n_gpu_layers,
    n_ctx=args.n_ctx,
    chat_format="chatml",
    verbose=False,
)
_lock = Lock()
logger.info("Model loaded successfully.")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{
        "id": args.model_alias, "object": "model",
        "created": int(time.time()), "owned_by": "local",
    }]}

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    stop = req.stop or ["<|im_end|>", "<|endoftext|>"]
    if req.stream:
        async def stream_gen():
            cid = f"cmpl-{uuid.uuid4().hex[:8]}"
            with _lock:
                for chunk in llm(
                    req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    stop=stop,
                    stream=True,
                ):
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        data = {
                            "id": cid, "object": "text_completion.chunk",
                            "created": int(time.time()), "model": args.model_alias,
                            "choices": [{"text": text, "index": 0, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        await asyncio.sleep(0)
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_gen(), media_type="text/event-stream")
    else:
        with _lock:
            result = llm(
                req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=stop,
            )
        text = result["choices"][0]["text"].strip()
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": args.model_alias,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            "usage": result.get("usage", {}),
        }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    stop = req.stop or ["<|im_end|>", "<|endoftext|>"]
    if req.stream:
        async def stream_gen():
            cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            with _lock:
                for chunk in llm.create_chat_completion(
                    messages=messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    stop=stop,
                    stream=True,
                ):
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        data = {
                            "id": cid, "object": "chat.completion.chunk",
                            "created": int(time.time()), "model": args.model_alias,
                            "choices": [{"delta": {"content": content}, "index": 0,
                                         "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        await asyncio.sleep(0)
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_gen(), media_type="text/event-stream")
    else:
        with _lock:
            result = llm.create_chat_completion(
                messages=messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=stop,
            )
        content = result["choices"][0]["message"]["content"].strip()
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": args.model_alias,
            "choices": [{"index": 0,
                          "message": {"role": "assistant", "content": content},
                          "finish_reason": "stop"}],
            "usage": result.get("usage", {}),
        }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")