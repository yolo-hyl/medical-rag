#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可扩展的文本标注框架
支持多种数据源、多种模型后端和高性能批量处理
"""

import json
import asyncio
import aiofiles
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnnotationConfig:
    """标注配置类"""
    model_base_url: str = "172.16.40.51:11434"
    model_backend: str = "ollama"  # ollama, vllm
    model_name: str = "qwen3:32b"
    temperature: float = 0.1
    max_concurrent: int = 5  # 最大并发数
    batch_size: int = 10  # 批处理大小
    max_retries: int = 3
    cache_enabled: bool = True
    output_format: str = "json"

@dataclass 
class DataItem:
    """数据项基类"""
    content: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AnnotationResult:
    """标注结果类"""
    original_data: DataItem
    annotations: Dict[str, Any]
    processing_time: float
    confidence: float = 1.0

# ==================== 抽象基类 ====================

class BaseDataLoader(ABC):
    """数据加载器基类"""
    
    @abstractmethod
    async def load_data(self, source: Union[str, Path]) -> AsyncGenerator[DataItem, None]:
        """异步加载数据"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """获取支持的格式"""
        pass

class BaseModelBackend(ABC):
    """模型后端基类"""
    
    def __init__(self, config: AnnotationConfig):
        self.config = config
    
    @abstractmethod
    async def initialize(self):
        """初始化模型"""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """生成文本"""
        pass
    
    @abstractmethod
    async def batch_generate(self, prompts: List[str]) -> List[str]:
        """批量生成"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """清理资源"""
        pass

class BaseAnnotator(ABC):
    """标注器基类"""
    
    def __init__(self, model_backend: BaseModelBackend, config: AnnotationConfig):
        self.model_backend = model_backend
        self.config = config
        self.cache = {} if config.cache_enabled else None
    
    @abstractmethod
    def create_prompt(self, data_item: DataItem) -> str:
        """创建提示词"""
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        """解析模型响应"""
        pass
    
    async def annotate_single(self, data_item: DataItem) -> AnnotationResult:
        """标注单个数据项"""
        start_time = time.time()
        
        # 检查缓存
        cache_key = hash(data_item.content) if self.cache is not None else None
        if cache_key and cache_key in self.cache:
            logger.info(f"Cache hit for item: {data_item.content[:50]}...")
            return self.cache[cache_key]
        
        # 创建提示词
        prompt = self.create_prompt(data_item)
        
        # 重试机制
        for attempt in range(self.config.max_retries):
            try:
                response = await self.model_backend.generate(prompt)
                annotations = self.parse_response(response)
                
                processing_time = time.time() - start_time
                result = AnnotationResult(
                    original_data=data_item,
                    annotations=annotations,
                    processing_time=processing_time
                )
                
                # 缓存结果
                if cache_key:
                    self.cache[cache_key] = result
                
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # 指数退避
    
    async def annotate_batch(self, data_items: List[DataItem]) -> List[AnnotationResult]:
        """批量标注"""
        # 分组批处理
        results = []
        for i in range(0, len(data_items), self.config.batch_size):
            batch = data_items[i:i + self.config.batch_size]
            
            # 并发处理批次
            semaphore = asyncio.Semaphore(self.config.max_concurrent)
            
            async def process_with_semaphore(item):
                async with semaphore:
                    return await self.annotate_single(item)
            
            batch_results = await asyncio.gather(
                *[process_with_semaphore(item) for item in batch],
                return_exceptions=True
            )
            
            # 处理异常
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process item {i+j}: {result}")
                    # 创建错误结果
                    error_result = AnnotationResult(
                        original_data=batch[j],
                        annotations={"error": str(result)},
                        processing_time=0,
                        confidence=0
                    )
                    results.append(error_result)
                else:
                    results.append(result)
                    
        return results

# ==================== 具体实现类 ====================

class QADataLoader(BaseDataLoader):
    """QA数据加载器"""
    
    async def load_data(self, source: Union[str, Path]) -> AsyncGenerator[DataItem, None]:
        if isinstance(source, str):
            # 处理JSON字符串或字典
            try:
                data = json.loads(source) if isinstance(source, str) else source
            except json.JSONDecodeError:
                data = source
        elif isinstance(source, Path):
            # 处理文件
            async with aiofiles.open(source, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
        else:
            data = source
            
        # 解析QA格式
        questions = data.get("questions", [])
        answers = data.get("answers", [])
        
        # 展开问题
        question = questions[0] if len(questions) != 1 else questions
        answer = answers[0] if len(answers) != 1 else answers
        
        flattened_questions = []
        for q_group in questions:
            if isinstance(q_group, list):
                flattened_questions.extend(q_group)
            else:
                flattened_questions.append(q_group)
        
        # 生成数据项
        for i, question in enumerate(flattened_questions):
            answer = answers[min(i, len(answers)-1)] if answers else ""
            yield DataItem(
                content=f"Q: {question}\nA: {answer}",
                metadata={"question": question, "answer": answer, "type": "qa"}
            )
    
    def get_supported_formats(self) -> List[str]:
        return ["json", "dict"]

class TxtDataLoader(BaseDataLoader):
    """文本文件数据加载器"""
    
    async def load_data(self, source: Union[str, Path]) -> AsyncGenerator[DataItem, None]:
        async with aiofiles.open(source, 'r', encoding='utf-8') as f:
            lines = await f.readlines()
            
        current_content = []
        line_number = 0
        
        for line in lines:
            line = line.strip()
            line_number += 1
            
            if line:  # 非空行
                current_content.append(line)
            else:  # 空行作为分隔符
                if current_content:
                    yield DataItem(
                        content='\n'.join(current_content),
                        metadata={"line_start": line_number - len(current_content), 
                                 "line_end": line_number - 1, "type": "text"}
                    )
                    current_content = []
        
        # 处理最后一段
        if current_content:
            yield DataItem(
                content='\n'.join(current_content),
                metadata={"line_start": line_number - len(current_content), 
                         "line_end": line_number, "type": "text"}
            )
    
    def get_supported_formats(self) -> List[str]:
        return ["txt", "text"]

class TermPairLoader(BaseDataLoader):
    """名词-解释对数据加载器"""
    
    async def load_data(self, source: Union[str, Path]) -> AsyncGenerator[DataItem, None]:
        if isinstance(source, (str, dict)):
            data = json.loads(source) if isinstance(source, str) else source
        else:
            async with aiofiles.open(source, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
        
        # 支持多种格式
        if "terms" in data:
            # 格式: {"terms": [{"term": "名词", "definition": "解释"}]}
            for term_data in data["terms"]:
                yield DataItem(
                    content=f"Term: {term_data['term']}\nDefinition: {term_data['definition']}",
                    metadata={"term": term_data["term"], "definition": term_data["definition"], "type": "term_pair"}
                )
        elif isinstance(data, dict):
            # 格式: {"名词1": "解释1", "名词2": "解释2"}
            for term, definition in data.items():
                yield DataItem(
                    content=f"Term: {term}\nDefinition: {definition}",
                    metadata={"term": term, "definition": definition, "type": "term_pair"}
                )
    
    def get_supported_formats(self) -> List[str]:
        return ["json", "dict"]

class OllamaBackend(BaseModelBackend):
    """Ollama模型后端"""
    
    def __init__(self, config: AnnotationConfig):
        super().__init__(config)
        self.llm = None
    
    async def initialize(self):
        from langchain_ollama import OllamaLLM
        self.llm = OllamaLLM(
            base_url=self.config.model_base_url,
            model=self.config.model_name,
            temperature=self.config.temperature
        )
        logger.info(f"Ollama backend initialized with model: {self.config.model_name}")
    
    async def generate(self, prompt: str) -> str:
        if not self.llm:
            await self.initialize()
        
        # 由于LangChain的Ollama可能不支持async，使用线程池
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, self.llm.invoke, prompt)
        return result
    
    async def batch_generate(self, prompts: List[str]) -> List[str]:
        # Ollama通常不支持原生批处理，使用并发调用
        tasks = [self.generate(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    async def cleanup(self):
        self.llm = None
        logger.info("Ollama backend cleaned up")

class VLLMBackend(BaseModelBackend):
    """VLLM模型后端（示例实现）"""
    
    def __init__(self, config: AnnotationConfig):
        super().__init__(config)
        self.client = None
    
    async def initialize(self):
        # 这里需要根据实际的VLLM客户端API进行实现
        logger.info(f"VLLM backend initialized with model: {self.config.model_name}")
        # self.client = VLLMClient(model=self.config.model_name)
        pass
    
    async def generate(self, prompt: str) -> str:
        # 示例实现
        # return await self.client.generate(prompt, temperature=self.config.temperature)
        raise NotImplementedError("VLLM backend not implemented")
    
    async def batch_generate(self, prompts: List[str]) -> List[str]:
        # VLLM原生支持批处理
        # return await self.client.batch_generate(prompts, temperature=self.config.temperature)
        raise NotImplementedError("VLLM backend not implemented")
    
    async def cleanup(self):
        if self.client:
            await self.client.close()
        logger.info("VLLM backend cleaned up")

class MedicalQAAnnotator(BaseAnnotator):
    """医学QA标注器"""
    
    def create_prompt(self, data_item: DataItem) -> str:
        question = data_item.metadata.get("question", "")
        answer = data_item.metadata.get("answer", "")
        
        return f"""
你是医学文本分类专家。请为以下QA对进行分类：

问题: {question}
答案: {answer}

科室分类(0-5)：
0-内科系统 1-外科系统 2-妇产儿科 3-五官感官 4-肿瘤影像 5-急诊综合

问题类别(0-7)：
0-疾病诊断与症状类 1-治疗方案类 2-药物与用药安全类 3-检查与化验类 4-预防保健类 5-特殊人群健康类 6-紧急情况与急救类 7-医学知识与科普类

请直接输出格式：
```json
{{
    "departments": [index1, index2...],
    "categories": [index1, index2...]
}}
```
"""
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        import re
        
        # 提取JSON部分
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.strip()
        
        try:
            parsed_data = json.loads(json_str)
            return {
                "departments": parsed_data.get("departments", []),
                "categories": parsed_data.get("categories", [])
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {"departments": [], "categories": [], "error": str(e)}

class TextClassificationAnnotator(BaseAnnotator):
    """通用文本分类标注器"""
    
    def __init__(self, model_backend: BaseModelBackend, config: AnnotationConfig, 
                 classification_schema: Dict[str, Any]):
        super().__init__(model_backend, config)
        self.classification_schema = classification_schema
    
    def create_prompt(self, data_item: DataItem) -> str:
        content = data_item.content
        
        # 动态构建分类选项
        schema_text = ""
        for category, options in self.classification_schema.items():
            schema_text += f"\n{category}:\n"
            for idx, option in enumerate(options):
                schema_text += f"{idx}-{option} "
        
        return f"""
请对以下文本进行分类：

文本内容: {content}

分类标准：{schema_text}

请输出JSON格式：
```json
{{
    {json.dumps({cat: ["index1", "index2"] for cat in self.classification_schema.keys()}, ensure_ascii=False)}
}}
```
"""
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        import re
        
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {cat: [] for cat in self.classification_schema.keys()}

# ==================== 框架主类 ====================

class AnnotationFramework:
    """标注框架主类"""
    
    def __init__(self, config: AnnotationConfig):
        self.config = config
        self.data_loaders: Dict[str, BaseDataLoader] = {}
        self.model_backend: Optional[BaseModelBackend] = None
        self.annotators: Dict[str, BaseAnnotator] = {}
        
        # 注册默认组件
        self._register_default_components()
    
    def _register_default_components(self):
        """注册默认组件"""
        # 注册数据加载器
        self.register_data_loader("qa", QADataLoader())
        self.register_data_loader("txt", TxtDataLoader())
        self.register_data_loader("term_pair", TermPairLoader())
        
        # 初始化模型后端
        if self.config.model_backend == "ollama":
            self.model_backend = OllamaBackend(self.config)
        elif self.config.model_backend == "vllm":
            self.model_backend = VLLMBackend(self.config)
        else:
            raise ValueError(f"Unsupported model backend: {self.config.model_backend}")
        
        # 注册标注器
        if self.model_backend:
            self.register_annotator("medical_qa", MedicalQAAnnotator(self.model_backend, self.config))
    
    def register_data_loader(self, name: str, loader: BaseDataLoader):
        """注册数据加载器"""
        self.data_loaders[name] = loader
        logger.info(f"Registered data loader: {name}")
    
    def register_annotator(self, name: str, annotator: BaseAnnotator):
        """注册标注器"""
        self.annotators[name] = annotator
        logger.info(f"Registered annotator: {name}")
    
    async def annotate(self, data_source: Union[str, Path], 
                      loader_type: str, annotator_type: str,
                      output_path: Optional[Path] = None) -> List[AnnotationResult]:
        """执行标注任务"""
        
        # 验证组件
        if loader_type not in self.data_loaders:
            raise ValueError(f"Unknown data loader: {loader_type}")
        if annotator_type not in self.annotators:
            raise ValueError(f"Unknown annotator: {annotator_type}")
        
        # 获取组件
        loader = self.data_loaders[loader_type]
        annotator = self.annotators[annotator_type]
        
        # 初始化模型后端
        await self.model_backend.initialize()
        
        try:
            # 加载数据
            logger.info(f"Loading data using {loader_type} loader...")
            data_items = []
            async for item in loader.load_data(data_source):
                data_items.append(item)
            
            logger.info(f"Loaded {len(data_items)} data items")
            
            # 执行标注
            logger.info(f"Starting annotation using {annotator_type} annotator...")
            start_time = time.time()
            
            results = await annotator.annotate_batch(data_items)
            
            total_time = time.time() - start_time
            logger.info(f"Annotation completed in {total_time:.2f}s")
            logger.info(f"Average time per item: {total_time/len(results):.2f}s")
            
            # 保存结果
            if output_path:
                await self._save_results(results, output_path)
            
            return results
            
        finally:
            # 清理资源
            await self.model_backend.cleanup()
    
    async def _save_results(self, results: List[AnnotationResult], output_path: Path):
        """保存结果"""
        output_data = []
        for result in results:
            item_data = {
                "content": result.original_data.content,
                "metadata": result.original_data.metadata,
                "annotations": result.annotations,
                "processing_time": result.processing_time,
                "confidence": result.confidence
            }
            output_data.append(item_data)
        
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(output_data, ensure_ascii=False, indent=2))
        
        logger.info(f"Results saved to: {output_path}")