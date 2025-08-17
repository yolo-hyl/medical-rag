from .DatasetLabeler import DatasetLabeler
from ...core.base.BaseClient import LLMClient
from ...config.data.annotator import LabelingConfig
from typing import Any, Dict, List, Optional, Union
import asyncio
import time, re, json

class QALabeler(DatasetLabeler):
    def __init__(self, llm_client: LLMClient, config: LabelingConfig, logger):
        super().__init__(llm_client, config, logger)
        self.system = config.system_prompt if config.system_prompt else None

    def sangle_response_get_label(self, response_text):
        """
        解析单条数据
        """
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # 如果没有代码块，尝试直接解析
            json_str = response_text.strip()
        return json.loads(json_str)
    
    async def label_single_sample(self, sample: Dict, retries: int = 0) -> Optional[Dict]:
        """标注单个样本（带重试）"""
        try:
            # 构造prompt
            question = sample.get("questions")[0][0]
            answer = sample.get("answers")[0]
            prompt = self.config.prompt_template.format(question=question, answer=answer)
            
            # 生成标签
            response_text = await self.llm_client.generate(prompt, system=self.system)
                
            parsed_data: dict = self.sangle_response_get_label(response_text)
            
            # 返回带标签的样本
            labeled_sample = sample.copy()
            labeled_sample["departments"] = parsed_data["departments"]
            labeled_sample["categories"] = parsed_data["categories"]
            labeled_sample["labeling_timestamp"] = time.time()
            
            return labeled_sample
            
        except Exception as e:
            if retries < self.config.max_retries:
                self.logger.warning(f"标注失败，重试 {retries + 1}/{self.config.max_retries}: {e}")
                await asyncio.sleep(self.config.retry_delay * (2 ** retries))  # 指数退避
                return await self.label_single_sample(sample, retries + 1)
            else:
                self.logger.error(f"标注最终失败: {e}")
                return None
            
    async def label_batch(self, batch: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """标注一批样本"""
        try:
            # 构造批量prompts
            prompts = []
            for sample in batch:
                question = sample.get("questions")[0][0]
                answer = sample.get("answers")[0]
                prompt = self.config.prompt_template.format(question=question, answer=answer)
                prompts.append(prompt)
            
            # 批量生成
            response_text = await self.llm_client.batch_generate(prompts, system=self.system)
            
            # 处理结果
            successful = []
            failed = []
            
            for sample, label in zip(batch, response_text):
                if label:  # 生成成功
                    try:
                        labeled_sample = sample.copy()
                        parsed_data: dict = self.sangle_response_get_label(label)
                        labeled_sample["departments"] = parsed_data["departments"]
                        labeled_sample["categories"] = parsed_data["categories"]
                        labeled_sample["labeling_timestamp"] = time.time()
                        successful.append(labeled_sample)
                    except Exception as e:
                        self.logger.error(f"解析标签失败: {e}")
                        failed.append(sample)
                else:  # 生成失败
                    failed.append(sample)
            
            return successful, failed
            
        except Exception as e:
            self.logger.error(f"批量标注失败: {e}")
            # 降级到逐个处理
            successful = []
            failed = []
            
            for sample in batch:
                result = await self.label_single_sample(sample)
                if result:
                    successful.append(result)
                else:
                    failed.append(sample)
            
            return successful, failed