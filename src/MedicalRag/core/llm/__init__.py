from .LocalClient import LocalModelClient
from .HttpClient import OllamaClient, OpenAICompatibleClient
from ..base.BaseClient import LLMClient
from typing import Dict, Any, List
from .VllmClient import VLLMClient

def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """创建LLM客户端工厂函数"""
    llm_config = config.get('llm_client', {})
    client_type = llm_config.get('type', 'ollama')
    
    client_map = {
        "ollama": OllamaClient,
        "vllm": VLLMClient,
        "openai": OpenAICompatibleClient,
        "local": LocalModelClient
    }
    
    if client_type not in client_map:
        raise ValueError(f"不支持的客户端类型: {client_type}")
    
    return client_map[client_type](llm_config)