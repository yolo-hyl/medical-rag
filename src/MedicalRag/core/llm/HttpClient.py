"""
更健壮的 Chat/LLM 客户端（多厂商），基于 httpx 异步请求，带超时/重试/网络代理，支持 system 指令
"""
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import logging
import math
import os
import json
import httpx

from MedicalRag.core.base.BaseClient import LLMClient  # 你原本的接口

# -----------------------
# 通用 HTTP 工具（重试/超时/代理）
# -----------------------

class _HTTPXClientManager:
    """
    维护一个 httpx.AsyncClient（长连接），提供带重试的 POST JSON 调用。
    """
    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        proxy: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        verify: bool = True,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        backoff_cap: float = 4.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.proxy = proxy
        self.verify = verify
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._headers = headers or {}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            proxy=self.proxy,
            verify=self.verify,
            headers=self._headers,
        )

    async def aclose(self):
        await self._client.aclose()

    async def post_json(
        self,
        path: str,
        json_data: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None,
        expected_status: Tuple[int, ...] = (200,),
    ) -> Dict[str, Any]:
        """
        发送 POST JSON，含指数退避重试。
        对网络错误/超时、5xx、429 进行重试。
        """
        url = path if path.startswith("/") else f"/{path}"
        headers = self._headers.copy()
        if extra_headers:
            headers.update(extra_headers)

        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(url, json=json_data, headers=headers)
                # 可重试状态码
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"Retryable status {resp.status_code}", request=resp.request, response=resp
                    )
                if resp.status_code not in expected_status:
                    # 非预期状态码，直接抛错（不重试）
                    text = resp.text
                    raise RuntimeError(f"HTTP {resp.status_code}: {text[:500]}")
                # 尝试解析 JSON
                try:
                    return resp.json()
                except Exception:
                    raise RuntimeError(f"Invalid JSON response: {resp.text[:500]}")
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                # 指数退避
                sleep_s = min(self.backoff_cap, self.backoff_base * (2 ** attempt))
                jitter = 0.1 * sleep_s
                await asyncio.sleep(sleep_s + (jitter * (2 * (attempt % 2) - 1)))
            except Exception as e:
                # 其它错误不重试
                raise
        # 重试多次仍失败
        assert last_exc is not None
        raise last_exc


# -----------------------
# Ollama 客户端（/api/generate）
# -----------------------

class OllamaClient(LLMClient):
    """
    Ollama 客户端（非流式），支持 system 指令，支持并发、超时、重试、代理。
    文档参考：https://github.com/jmorganca/ollama/blob/main/docs/api.md
    """
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config['model_name']
        ollama_config = config.get('ollama', {})
        base_url = ollama_config.get('base_url', 'http://localhost:11434')
        timeout = float(ollama_config.get('timeout', 60))
        self.max_concurrent = int(ollama_config.get('max_concurrent', 8))
        self.manager = _HTTPXClientManager(
            base_url=base_url,
            timeout=timeout,
            proxy=ollama_config.get('proxy'),
            verify=ollama_config.get('verify', True),
            max_retries=ollama_config.get('max_retries', 3),
            backoff_base=ollama_config.get('backoff_base', 0.5),
            backoff_cap=ollama_config.get('backoff_cap', 4.0),
        )

    async def aclose(self):
        await self.manager.aclose()

    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        单次生成（非流式）。Ollama /api/generate 支持 'system' 字段。
        """
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            data["system"] = system

        try:
            result = await self.manager.post_json("api/generate", data)
            return (result.get("response") or "").strip()
        except Exception as e:
            logging.error(f"Ollama 生成失败: {e}")
            raise

    async def batch_generate(self, prompts: List[str], system: Optional[str] = None) -> List[str]:
        """
        并发批量生成。
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _task(p: str):
            async with semaphore:
                return await self.generate(p, system=system)

        return await asyncio.gather(*[_task(p) for p in prompts])


# -----------------------
# OpenAI 兼容 REST 客户端（/v1/chat/completions）
# -----------------------

class OpenAICompatibleClient(LLMClient):
    """
    兼容 OpenAI Chat Completions 的 REST 客户端（非 SDK）。
    - 默认 base_url: https://api.openai.com/v1
    - 接口路径: /chat/completions
    - 支持 system 指令
    - 支持 max_tokens/temperature/top_p 等常用参数
    - 支持并发、超时、重试、代理
    你也可以把 base_url 换成兼容服务（如自建代理网关）。
    """
    def __init__(self, config: Dict[str, Any]):
        openai_config = config.get('openai', {})

        self.model_name = openai_config.get('model_name', 'gpt-4o-mini')
        self.max_tokens = int(openai_config.get('max_tokens', 512))
        self.temperature = float(openai_config.get('temperature', 0.2))
        self.top_p = float(openai_config.get('top_p', 1.0))
        self.max_concurrent = int(openai_config.get('max_concurrent', 5))

        # 认证 & 端点
        api_key = openai_config.get('api_key') or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.warning("未提供 OPENAI_API_KEY，若目标服务需要鉴权将会失败。")
        base_url = openai_config.get('base_url', 'https://api.openai.com/v1')
        timeout = float(openai_config.get('timeout', 60))

        default_headers = {
            "Authorization": f"Bearer {api_key}" if api_key else "",
            "Content-Type": "application/json",
        }
        # 可选组织/项目头
        if org := openai_config.get("organization"):
            default_headers["OpenAI-Organization"] = org
        if project := openai_config.get("project"):
            default_headers["OpenAI-Project"] = project

        self.manager = _HTTPXClientManager(
            base_url=base_url,
            timeout=timeout,
            proxy=openai_config.get('proxy'),
            verify=openai_config.get('verify', True),
            max_retries=openai_config.get('max_retries', 3),
            backoff_base=openai_config.get('backoff_base', 0.5),
            backoff_cap=openai_config.get('backoff_cap', 4.0),
            headers=default_headers,
        )

        # 某些“兼容”服务把 chat 路径放在别名上，这里允许覆盖
        self._chat_path = openai_config.get("chat_path", "chat/completions")

    async def aclose(self):
        await self.manager.aclose()

    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        单次生成（非流式）。使用 Chat Completions 语义。
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }

        try:
            result = await self.manager.post_json(self._chat_path, payload)
            # 兼容多种返回结构（标准为 choices[0].message.content）
            choices = result.get("choices") or []
            if not choices:
                # 某些兼容实现可能返回 "error"
                err = result.get("error") or {}
                msg = err.get("message") or str(result)[:500]
                raise RuntimeError(f"无可用生成结果：{msg}")
            msg = choices[0].get("message") or {}
            content = (msg.get("content") or "").strip()
            return content
        except Exception as e:
            logging.error(f"OpenAI 兼容生成失败: {e}")
            raise

    async def batch_generate(self, prompts: List[str], system: Optional[str] = None) -> List[str]:
        """
        并发批量生成。
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _task(p: str):
            async with semaphore:
                return await self.generate(p, system=system)

        return await asyncio.gather(*[_task(p) for p in prompts])
