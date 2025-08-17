from typing import Dict, Any, List
import asyncio
from MedicalRag.core.base.BaseClient import LLMClient
import requests
import logging

class VLLMClient(LLMClient):
    """vLLM客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config['model_name']
        vllm_config = config.get('vllm', {})
        
        self.base_url = vllm_config.get('base_url', 'http://localhost:8000')
        self.timeout = vllm_config.get('timeout', 60)
        self.max_tokens = vllm_config.get('max_tokens', 512)
        self.temperature = vllm_config.get('temperature', 0.1)
        self.session = requests.Session()
    
    async def generate(self, prompt: str) -> str:
        """单个文本生成"""
        url = f"{self.base_url}/v1/completions"
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = self.session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"].strip()
        except Exception as e:
            logging.error(f"vLLM生成失败: {e}")
            raise
    
    async def batch_generate(self, prompts: List[str]) -> List[str]:
        """批量生成 - vLLM原生支持批处理"""
        url = f"{self.base_url}/v1/completions"
        
        try:
            results = []
            for prompt in prompts:
                data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
                response = self.session.post(url, json=data, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                results.append(result["choices"][0]["text"].strip())
            return results
        except Exception as e:
            logging.error(f"vLLM批量生成失败: {e}")
            return await asyncio.gather(*[self.generate(prompt) for prompt in prompts])
        
    async def embedding(self, prompt: str) -> List[float]:
        # TODO 编码
        raise "没实现"
        
        
    async def batch_embedding(self, prompts: List[str]) -> List[List[float]]:
        # TODO 编码
        raise "没实现"
