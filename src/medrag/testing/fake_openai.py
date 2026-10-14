"""本地的假 OpenAI 兼容服务：不需要密钥和网络，就能跑通入库、检索、RAG、Agent 的完整链路

    with FakeOpenAIServer() as server:            # 在后台线程里监听一个空闲端口
        cfg.llm.base_url = server.base_url        # http://127.0.0.1:<port>/v1
        ...

也可以单独起一个进程给脚本用：python -m medrag.testing.fake_openai --port 8999

- /v1/embeddings：按字符二元组哈希生成确定性向量（L2 归一化）。字面上相近的文本向量也相近，
  所以检索结果有意义，但它不是语义模型，只能用来验证链路和做新旧实现的一致性对比。
  同时支持 encoding_format=float / base64（openai SDK 默认 base64）。
- /v1/chat/completions：
    带 tools 时返回对第一个工具的调用，参数按工具的 JSON Schema 填（字符串填用户最后一句话）；
    提示词要求结构化输出时（出现 ask_signal / rewritten_queries / need_search 等字段名）返回对应 JSON；
    事实校验（仅输出Y或N）返回 Y；其余返回固定回答。支持 stream=True。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

DEFAULT_DIM = 1024
DEFAULT_REPLY = "这是一个用于测试的回答。"


def fake_embedding(text: str, dim: int = DEFAULT_DIM) -> List[float]:
    vec = [0.0] * dim
    text = text or ""
    grams = [text[i:i + 2] for i in range(max(1, len(text) - 1))] or [""]
    for g in grams:
        h = hashlib.md5(g.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _last_user_text(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):  # 多段内容
                return "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return content or ""
    return ""


def _fill_args(schema: Dict[str, Any], text: str) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for name, prop in (schema.get("properties") or {}).items():
        typ = prop.get("type")
        if typ == "string":
            args[name] = text[:200]
        elif typ == "integer":
            args[name] = prop.get("default", 3)
        elif typ == "number":
            args[name] = prop.get("default", 1.0)
        elif typ == "boolean":
            args[name] = prop.get("default", False)
    return args


def chat_reply(body: Dict[str, Any]) -> Dict[str, Any]:
    """返回 assistant 消息（content 或 tool_calls）"""
    messages = body.get("messages") or []
    all_text = "\n".join(str(m.get("content") or "") for m in messages)
    user_text = _last_user_text(messages)

    tools = body.get("tools") or []
    if tools:
        fn = tools[0]["function"]
        args = _fill_args(fn.get("parameters") or {}, user_text)
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"call_{hashlib.md5(user_text.encode()).hexdigest()[:8]}",
            "type": "function",
            "function": {"name": fn["name"], "arguments": json.dumps(args, ensure_ascii=False)},
        }]}

    if "ask_signal" in all_text:
        content = json.dumps({"ask_signal": False, "questions": []}, ensure_ascii=False)
    elif "rewritten_queries" in all_text:
        content = json.dumps({"rewritten_queries": [user_text.strip()[:50] or "医学问题"]}, ensure_ascii=False)
    elif "need_search" in all_text:
        content = json.dumps({"need_search": False, "search_query": "", "remain_doc_index": []}, ensure_ascii=False)
    elif "仅输出Y或N" in all_text:
        content = "Y"
    else:
        content = DEFAULT_REPLY
    return {"role": "assistant", "content": content}


class _Handler(BaseHTTPRequestHandler):
    server_version = "FakeOpenAI/1.0"

    def log_message(self, fmt, *args):  # 安静一点，测试输出里不需要访问日志
        pass

    def _json(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append((self.path, body))  # type: ignore[attr-defined]
        if self.path.endswith("/embeddings"):
            self._embeddings(body)
        elif self.path.endswith("/chat/completions"):
            self._chat(body)
        else:
            self._json(404, {"error": {"message": f"未实现的接口 {self.path}"}})

    def _embeddings(self, body: Dict[str, Any]) -> None:
        inputs = body.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        dim = int(body.get("dimensions") or self.server.default_dim)  # type: ignore[attr-defined]
        b64 = body.get("encoding_format") == "base64"
        data = []
        for i, text in enumerate(inputs):
            vec = fake_embedding(str(text), dim)
            emb: Any = base64.b64encode(struct.pack(f"<{dim}f", *vec)).decode() if b64 else vec
            data.append({"object": "embedding", "index": i, "embedding": emb})
        tokens = sum(len(str(t)) for t in inputs)
        self._json(200, {"object": "list", "data": data, "model": body.get("model", "fake"),
                         "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})

    def _chat(self, body: Dict[str, Any]) -> None:
        msg = chat_reply(body)
        prompt_tokens = sum(len(str(m.get("content") or "")) for m in body.get("messages") or [])
        completion_tokens = len(msg.get("content") or "") or 1
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens}
        base = {"id": "chatcmpl-fake", "created": int(time.time()), "model": body.get("model", "fake")}
        if not body.get("stream"):
            self._json(200, {**base, "object": "chat.completion", "usage": usage, "choices": [
                {"index": 0, "message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}]})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(chunk: Dict[str, Any]) -> None:
            self.wfile.write(f"data: {json.dumps({**base, 'object': 'chat.completion.chunk', **chunk}, ensure_ascii=False)}\n\n".encode())

        if msg.get("tool_calls"):
            call = dict(msg["tool_calls"][0], index=0)
            send({"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [call]}, "finish_reason": None}]})
            send({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        else:
            content = msg["content"]
            for i in range(0, len(content), 4):
                send({"choices": [{"index": 0, "delta": {"content": content[i:i + 4]}, "finish_reason": None}]})
            send({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        if (body.get("stream_options") or {}).get("include_usage"):
            send({"choices": [], "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")


class FakeOpenAIServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, default_dim: int = DEFAULT_DIM):
        self.httpd = ThreadingHTTPServer((host, port), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.requests = []          # type: ignore[attr-defined]  便于断言收到了哪些请求
        self.httpd.default_dim = default_dim  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def requests(self) -> List[tuple]:
        return self.httpd.requests  # type: ignore[attr-defined]

    def start(self) -> "FakeOpenAIServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self) -> "FakeOpenAIServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="本地假 OpenAI 兼容服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8999)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    args = parser.parse_args()
    server = FakeOpenAIServer(args.host, args.port, args.dim)
    print(f"假 OpenAI 服务：{server.base_url}（Ctrl+C 退出）")
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.httpd.server_close()


if __name__ == "__main__":
    main()
