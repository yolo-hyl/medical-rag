"""
加载器的基类
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)



class BaseLoader(ABC):
    """JSONL格式QA数据加载器"""
    
    def __init__(self):
        pass
    
    @abstractmethod
    def load_file(self, file_path: Path):
        pass
    
    @abstractmethod
    def load_multiple_files(self, file_paths: List[Path]):
        pass
    
    @abstractmethod
    def load_directory(
        self, 
        directory: Path, 
        pattern: str,
        recursive: bool = True
    ):
        pass
    
    
    @abstractmethod
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def _log_loading_stats(self):
        pass
    
    @abstractmethod
    def get_stats(self):
        pass
    