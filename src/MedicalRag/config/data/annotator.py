from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import importlib
import yaml

def get_prompt_template(config: Dict[str, Any], sys: bool) -> str:
    """获取prompt模板"""
    prompt_config = config.get('prompt', {})
    
    if prompt_config.get('use_template_from_config', False):
        return prompt_config.get('template', '')
    else:
        # 从模块导入
        if sys:
            module_name = prompt_config.get('system_prompt_module', None)
            variable_name = prompt_config.get('system_prompt_variable', None)
            if module_name == None or module_name == "" or variable_name == None or variable_name == "":
                return None
        else:
            module_name = prompt_config.get('template_module', 'MedicalRag.config.prompts')
            variable_name = prompt_config.get('template_variable', 'ANNOTATION_PROMPT')
        try:
            module = importlib.import_module(module_name)
            return getattr(module, variable_name)
        except Exception as e:
            raise ValueError(f"无法从 {module_name} 导入 {variable_name}: {e}")
        
def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ValueError(f"无法加载配置文件 {config_path}: {e}")
    
    
@dataclass
class LabelingConfig:
    """标注配置类"""
    # 基本配置
    model_name: str
    batch_size: int = 8
    max_retries: int = 3
    retry_delay: float = 1.0
    checkpoint_interval: int = 100
    output_dir: str = "./output"
    prompt_template: str = ""
    system_prompt: str = None
    max_workers: int = 4
    resume_from_checkpoint: bool = True
    proxy: str = None
    
    # 输出配置
    output_formats: List[str] = None
    checkpoint_file: str = "checkpoint.json"
    failed_file: str = "failed_samples.json"
    results_file: str = "labeled_dataset.json"
    dataset_dir: str = "labeled_dataset"
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LabelingConfig':
        """从配置字典创建实例"""
        labeling_config = config.get('labeling', {})
        llm_config = config.get('llm_client', {})
        output_config = config.get('output', {})
        
        return cls(
            model_name=llm_config.get('model_name', 'qwen3:32b'),
            batch_size=labeling_config.get('batch_size', 8),
            max_retries=labeling_config.get('max_retries', 3),
            retry_delay=labeling_config.get('retry_delay', 1.0),
            checkpoint_interval=labeling_config.get('checkpoint_interval', 100),
            output_dir=labeling_config.get('output_dir', './output'),
            prompt_template=get_prompt_template(config, sys=False),
            system_prompt=get_prompt_template(config, sys=True),
            proxy=llm_config.get('proxy', None),
            max_workers=labeling_config.get('max_workers', 4),
            resume_from_checkpoint=labeling_config.get('resume_from_checkpoint', True),
            output_formats=output_config.get('formats', ['json']),
            checkpoint_file=output_config.get('checkpoint_file', 'checkpoint.json'),
            failed_file=output_config.get('failed_file', 'failed_samples.json'),
            results_file=output_config.get('results_file', 'labeled_dataset.json'),
            dataset_dir=output_config.get('dataset_dir', 'labeled_dataset')
        )