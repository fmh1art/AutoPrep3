import argparse
import logging

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.module.cached_token_server import CachedTokenServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cached Token Server")

server: CachedTokenServer | None = None


class QueryAndCacheRequest(BaseModel):
    input_messages: list[dict]
    response_msg: dict | str | None = None


class ComputeRequest(BaseModel):
    input_messages: list[dict]


class CacheRequest(BaseModel):
    input_messages: list[dict]
    response_msg: dict | str | None = None


class CachedTokenResponse(BaseModel):
    cached_tokens: int


class StatusResponse(BaseModel):
    size: int
    max_records: int
    token_model: str


@app.post("/query_and_cache", response_model=CachedTokenResponse)
def query_and_cache(req: QueryAndCacheRequest):
    cached_tokens = server.query_and_cache(req.input_messages, req.response_msg)
    return CachedTokenResponse(cached_tokens=cached_tokens)


@app.post("/compute", response_model=CachedTokenResponse)
def compute(req: ComputeRequest):
    cached_tokens = server.compute_cached_tokens(req.input_messages)
    return CachedTokenResponse(cached_tokens=cached_tokens)


@app.post("/cache")
def cache(req: CacheRequest):
    server.cache_conversation(req.input_messages, req.response_msg)
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
def status():
    return StatusResponse(
        size=server.size,
        max_records=server.max_records,
        token_model=server.token_model,
    )


@app.get("/health")
def health():
    return {"status": "healthy"}


def main():
    parser = argparse.ArgumentParser(description="Cached Token HTTP Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18731)
    parser.add_argument("--max-records", type=int, default=10000)
    parser.add_argument("--token-model", type=str, default="cl100k_base")
    args = parser.parse_args()

    global server
    server = CachedTokenServer(
        max_records=args.max_records,
        token_model=args.token_model,
    )

    logger.info(
        f"Starting Cached Token Server on {args.host}:{args.port} "
        f"(max_records={args.max_records}, token_model={args.token_model})"
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
