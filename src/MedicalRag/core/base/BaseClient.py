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