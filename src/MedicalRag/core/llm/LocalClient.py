import logging
from typing import Any, Dict, List, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..base.BaseClient import LLMClient

class LocalModelClient(LLMClient):
    """本地模型客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config['model_name']
        local_config = config.get('local', {})
        
        device = local_config.get('device', 'auto')
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        torch_dtype = local_config.get('torch_dtype', 'float16')
        self.torch_dtype = torch.float16 if torch_dtype == 'float16' else torch.float32
        
        self.max_new_tokens = local_config.get('max_new_tokens', 512)
        self.temperature = local_config.get('temperature', 0.1)
        self.do_sample = local_config.get('do_sample', True)
        
        logging.info(f"加载本地模型: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    async def generate(self, prompt: str) -> str:
        """单个文本生成"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()
    
    async def batch_generate(self, prompts: List[str]) -> List[str]:
        """批量生成 - 使用模型的批处理能力"""
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        results = []
        for i, output in enumerate(outputs):
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            original_prompt = prompts[i]
            result = generated_text[len(original_prompt):].strip()
            results.append(result)
        
        return results