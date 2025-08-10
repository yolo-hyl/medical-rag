"""
Chat/LLM 客户端（可多厂商），超时/重试/网络代理
"""
"""
Ollama LLM客户端实现，支持思考模型的特殊标签处理
"""
import re
import json
import httpx
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from MedicalRag.core.base.BaseClient import LLMHttpClient

class OllamaClient(LLMHttpClient):
    """Ollama客户端，处理与本地Ollama服务的交互"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:32b",
        timeout: float = 60.0,
        thinking_model: bool = True,
        url: str = "/api/chat"
    ):
        super().__init__(
            base_url = base_url,
            model = model,
            timeout = timeout,
            thinking_model = thinking_model,
            url = url
        )
    
    def health_check(self) -> bool:
        """检查Ollama服务健康状态"""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> list[str]:
        """列出可用模型"""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except:
            pass
        return []
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'client'):
            self.client.close()


class AsyncOllamaClient:
    """异步Ollama客户端"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:32b",
        timeout: float = 60.0
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def chat(
        self, 
        messages: list[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """异步对话"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code}")