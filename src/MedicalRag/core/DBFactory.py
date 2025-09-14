# kb_factory.py
import os, atexit
from functools import lru_cache
from .KnowledgeBase import MedicalHybridKnowledgeBase
from ..config.models import AppConfig
import json

@lru_cache(maxsize=None)
def _kb_singleton(pid: int, config_str: str):
    """按 (pid, 配置) 缓存，保证每个进程只创建一次"""
    config_dict = json.loads(config_str)
    config = AppConfig.model_validate(config_dict)
    kb = MedicalHybridKnowledgeBase(config)
    # 进程退出时优雅关闭
    atexit.register(lambda: getattr(kb.client, "close", lambda: None)())
    return kb

def get_kb(config: dict):
    return _kb_singleton(os.getpid(), json.dumps(config, sort_keys=True))
