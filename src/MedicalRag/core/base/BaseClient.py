"""
LLM的客户端
"""
from typing import List, Optional
from abc import ABC, abstractmethod

class LLMClient(ABC):
    """LLM客户端抽象基类"""
    
    @abstractmethod
    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """生成文本"""
        pass
    
    @abstractmethod
    async def batch_generate(self, prompts: List[str], system: Optional[str] = None) -> List[str]:
        """批量生成文本"""
        pass
    
    @abstractmethod
    async def embedding(self, prompt: str) -> List[float]:
        """单条数据嵌入"""
        pass
    
    @abstractmethod
    async def batch_embedding(self, prompts: List[str]) -> List[List[float]]:
        """批量数据嵌入"""
        pass