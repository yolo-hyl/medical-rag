"""
LLM的客户端
"""

import re
import json
import httpx
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field, HttpUrl
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from abc import ABC, abstractmethod

class LLMLocalModelClient:
    def __init__(self) -> None:
        pass
    

class LLMHttpClientCfg(BaseModel):
    base_url: HttpUrl = Field("http://localhost:11434", description="LLM API 基础地址")
    model: str = Field("qwen3:32b", description="模型名称")
    timeout: float = Field(60.0, description="请求超时时间（秒）")
    thinking_model: bool = Field(True, description="是否开启thinking模式解析")
    url: str = Field("/api/chat", description="API 路径")
    proxies: Optional[Union[str, Dict[str, str]]] = Field(
        None, description="HTTP/HTTPS 代理设置"
    )

class LLMHttpClient(ABC):
    def __init__(
        self,
        cfg: LLMHttpClientCfg
    ) -> None:
        self.base_url = cfg.base_url.rstrip('/')
        self.model =cfg.model
        self.timeout = cfg.timeout
        self.client = httpx.Client(
            timeout=cfg.timeout,
            proxies=cfg.proxies
        )
        self.thinking_model = cfg.thinking_model
        self.url = cfg.url
    
    def _extract_thinking_content(self, text: str) -> tuple[str, str]:
        """
        提取思考模型的thinking标签内容和最终回答
        
        Args:
            text: 原始响应文本
            
        Returns:
            tuple: (thinking_content, final_answer)
        """
        # 匹配 <think>...</think> 标签
        think_pattern = r'<think>(.*?)</think>\s*\n\n'
        think_match = re.search(think_pattern, text, re.DOTALL)
        
        if think_match:
            thinking_content = think_match.group(1).strip()
            # 移除thinking部分，获取最终答案
            final_answer = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
        else:
            thinking_content = ""
            final_answer = text.strip()
            
        return thinking_content, final_answer
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def chat(
        self, 
        messages: list[Dict[str, str]], 
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        与模型进行对话
        
        Args:
            messages: 对话消息列表
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            响应结果
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        try:
            response = self.client.post(
                str(self.base_url + self.url),
                json=payload
            )
            response.raise_for_status()
            
            if stream:
                return response
            else:
                result = response.json()
                return result
                
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code} - {e.response.text}")
        
    def generate_completion(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成文本补全
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            extract_thinking: 是否提取thinking标签内容
            **kwargs: 其他参数
            
        Returns:
            包含响应和thinking内容的字典
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        result = self.chat(messages, **kwargs)
        
        if "message" in result and "content" in result["message"]:
            content = result["message"]["content"]
            
            if self.thinking_model:
                thinking, answer = self._extract_thinking_content(content)
                return {
                    "thinking": thinking,
                    "answer": answer,
                    "raw_content": content,
                    "model": self.model,
                    "done": result.get("done", True)
                }
            else:
                return {
                    "answer": content,
                    "raw_content": content,
                    "model": self.model,
                    "done": result.get("done", True)
                }
        else:
            raise RuntimeError(f"Unexpected response format: {result}")