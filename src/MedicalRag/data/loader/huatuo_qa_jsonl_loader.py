"""
JSON/JSONL 加载（jq路径到 content/metadata）
"""
"""
JSONL格式QA数据加载器
"""
import json
import logging
from pathlib import Path
from typing import List, Iterator, Optional, Dict, Any
from dataclasses import dataclass
from MedicalRag.core.base.BaseLoader import BaseLoader
from MedicalRag.schemas.metadata import JSONLRecord


logger = logging.getLogger(__name__)


@dataclass
class LoaderStats:
    """加载器统计信息"""
    total_lines: int = 0
    valid_lines: int = 0
    invalid_lines: int = 0
    total_qa_pairs: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class JSONLLoader(BaseLoader):
    """JSONL格式QA数据加载器"""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        初始化加载器
        
        Args:
            encoding: 文件编码
        """
        self.encoding = encoding
        self.stats = LoaderStats()
    
    def load_file(self, file_path: Path) -> List[tuple[str, str]]:
        """
        加载JSONL文件并返回QA对列表
        
        Args:
            file_path: JSONL文件路径
            
        Returns:
            QA对列表 [(question, answer), ...]
        """
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if not file_path.suffix.lower() in ['.jsonl', '.json']:
            logger.warning(f"文件扩展名不是.jsonl或.json: {file_path}")
        
        logger.info(f"开始加载JSONL文件: {file_path}")
        
        qa_pairs = []
        self.stats = LoaderStats()
        
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                for line_num, line in enumerate(f, 1):
                    self.stats.total_lines += 1
                    line = line.strip()
                    
                    # 跳过空行
                    if not line:
                        continue
                    
                    try:
                        # 解析JSON行
                        json_data = json.loads(line)
                        record = JSONLRecord(**json_data)
                        
                        # 转换为QA对
                        pairs = record.to_qa_pairs()
                        qa_pairs.extend(pairs)
                        
                        self.stats.valid_lines += 1
                        self.stats.total_qa_pairs += len(pairs)
                        
                        logger.debug(f"第{line_num}行: 解析出{len(pairs)}个QA对")
                        
                    except json.JSONDecodeError as e:
                        error_msg = f"第{line_num}行JSON解析错误: {e}"
                        logger.error(error_msg)
                        self.stats.errors.append(error_msg)
                        self.stats.invalid_lines += 1
                        
                    except Exception as e:
                        error_msg = f"第{line_num}行处理错误: {e}"
                        logger.error(error_msg)
                        self.stats.errors.append(error_msg)
                        self.stats.invalid_lines += 1
        
        except Exception as e:
            error_msg = f"文件读取错误: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self._log_loading_stats()
        return qa_pairs
    
    def load_multiple_files(self, file_paths: List[Path]) -> List[tuple[str, str]]:
        """
        加载多个JSONL文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            合并的QA对列表
        """
        all_qa_pairs = []
        
        for file_path in file_paths:
            try:
                qa_pairs = self.load_file(file_path)
                all_qa_pairs.extend(qa_pairs)
                logger.info(f"从{file_path}加载了{len(qa_pairs)}个QA对")
            except Exception as e:
                logger.error(f"加载文件{file_path}失败: {e}")
                continue
        
        logger.info(f"总共加载了{len(all_qa_pairs)}个QA对")
        return all_qa_pairs
    
    def load_directory(
        self, 
        directory: Path, 
        pattern: str = "*.jsonl",
        recursive: bool = True
    ) -> List[tuple[str, str]]:
        """
        加载目录下的所有JSONL文件
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            recursive: 是否递归搜索子目录
            
        Returns:
            合并的QA对列表
        """
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"目录不存在或不是目录: {directory}")
        
        # 查找匹配的文件
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        if not files:
            logger.warning(f"在目录{directory}中未找到匹配{pattern}的文件")
            return []
        
        logger.info(f"在目录{directory}中找到{len(files)}个匹配文件")
        return self.load_multiple_files(files)
    
    def stream_load(self, file_path: Path) -> Iterator[tuple[str, str]]:
        """
        流式加载JSONL文件，逐个返回QA对
        
        Args:
            file_path: JSONL文件路径
            
        Yields:
            QA对 (question, answer)
        """
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        logger.info(f"开始流式加载JSONL文件: {file_path}")
        
        with open(file_path, 'r', encoding=self.encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    json_data = json.loads(line)
                    record = JSONLRecord(**json_data)
                    
                    # 逐个返回QA对
                    for qa_pair in record.to_qa_pairs():
                        yield qa_pair
                        
                except Exception as e:
                    logger.error(f"第{line_num}行处理错误: {e}")
                    continue
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """
        验证JSONL文件格式
        
        Args:
            file_path: JSONL文件路径
            
        Returns:
            验证结果字典
        """
        validation_result = {
            "is_valid": True,
            "total_lines": 0,
            "valid_lines": 0,
            "invalid_lines": 0,
            "errors": [],
            "sample_records": []
        }
        
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                for line_num, line in enumerate(f, 1):
                    validation_result["total_lines"] += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    try:
                        json_data = json.loads(line)
                        record = JSONLRecord(**json_data)
                        validation_result["valid_lines"] += 1
                        
                        # 保存前几个样本记录
                        if len(validation_result["sample_records"]) < 3:
                            validation_result["sample_records"].append({
                                "line_num": line_num,
                                "questions": record.questions,
                                "answers": record.answers,
                                "qa_pairs_count": len(record.to_qa_pairs())
                            })
                            
                    except Exception as e:
                        validation_result["invalid_lines"] += 1
                        validation_result["errors"].append(f"第{line_num}行: {e}")
                        validation_result["is_valid"] = False
        
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"文件读取错误: {e}")
        
        return validation_result
    
    def _log_loading_stats(self):
        """输出加载统计信息"""
        stats = self.stats
        logger.info("=== JSONL加载统计 ===")
        logger.info(f"总行数: {stats.total_lines}")
        logger.info(f"有效行数: {stats.valid_lines}")
        logger.info(f"无效行数: {stats.invalid_lines}")
        logger.info(f"总QA对数: {stats.total_qa_pairs}")
        
        if stats.invalid_lines > 0:
            logger.warning(f"存在{stats.invalid_lines}行无效数据")
            for error in stats.errors[:5]:  # 只显示前5个错误
                logger.warning(f"  {error}")
            if len(stats.errors) > 5:
                logger.warning(f"  ... 还有{len(stats.errors) - 5}个错误")
    
    def get_stats(self) -> LoaderStats:
        """获取加载统计信息"""
        return self.stats
    
    def export_qa_pairs_to_jsonl(
        self, 
        qa_pairs: List[tuple[str, str]], 
        output_path: Path,
        batch_size: int = 1000
    ):
        """
        将QA对导出为JSONL格式
        
        Args:
            qa_pairs: QA对列表
            output_path: 输出文件路径
            batch_size: 批处理大小
        """
        logger.info(f"导出{len(qa_pairs)}个QA对到: {output_path}")
        
        with open(output_path, 'w', encoding=self.encoding) as f:
            for i, (question, answer) in enumerate(qa_pairs):
                record = {
                    "questions": [question],
                    "answers": [answer]
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                if (i + 1) % batch_size == 0:
                    logger.info(f"已导出 {i + 1}/{len(qa_pairs)} 个QA对")
        
        logger.info(f"导出完成: {output_path}")