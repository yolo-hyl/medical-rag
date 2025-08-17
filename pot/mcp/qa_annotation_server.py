"""
QA数据标注MCP服务器
使用MCP协议提供数据标注工具，支持医疗QA数据的智能分类
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass

import mcp.types as types
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qa_annotation_server")

@dataclass
class QAItem:
    """QA数据项"""
    question: str
    answer: str
    departments: List[int]
    categories: List[int]

class QAAnnotationServer:
    """QA数据标注服务器"""
    
    def __init__(self):
        self.server = Server("qa-annotation-server")
        self.department_mapping = {
            0: "内科系统", 1: "外科系统", 2: "妇产与儿科", 3: "五官及感官系统", 4: "肿瘤与影像相关", 5: "急诊与综合科室"
        }
        
        self.category_mapping = {
            0: "疾病诊断与症状类", 1: "治疗方案类", 2: "药物与用药安全类", 3: "检查与化验类：各种检查方法、化验指标、影像学检查等",
            4: "预防与保健类", 5: "特殊人群健康类", 6: "紧急情况与急救类",
            7: "医学知识与科普类"
        }
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """设置MCP处理器"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """列出可用工具"""
            return [
                types.Tool(
                    name="annotate_qa_data",
                    description="对医疗QA数据进行智能标注，包括科室分类和问题类别分类",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "qa_data": {
                                "type": "object",
                                "description": "原始QA数据",
                                "properties": {
                                    "questions": {
                                        "type": "array",
                                        "description": "问题列表",
                                        "items": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "answers": {
                                        "type": "array",
                                        "description": "答案列表",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["questions", "answers"]
                            }
                        },
                        "required": ["qa_data"]
                    }
                ),
                types.Tool(
                    name="get_classification_schema",
                    description="获取分类体系的详细说明",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False
                    }
                ),
                types.Tool(
                    name="validate_annotation",
                    description="验证标注结果的格式和内容",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "annotated_data": {
                                "type": "array",
                                "description": "标注后的数据",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string"},
                                        "answer": {"type": "string"},
                                        "departments": {"type": "array", "items": {"type": "integer"}},
                                        "categories": {"type": "array", "items": {"type": "integer"}}
                                    }
                                }
                            }
                        },
                        "required": ["annotated_data"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict | None
        ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """处理工具调用"""
            
            if name == "get_classification_schema":
                return await self._get_classification_schema()
            elif name == "validate_annotation":
                return await self._validate_annotation(arguments)
            else:
                raise ValueError(f"未知工具: {name}")
    
    def _generate_classification_prompt(self, question: str, answer: str) -> str:
        """生成分类提示"""
        dept_descriptions = "\n".join([f"- {k}: {v}" for k, v in self.department_mapping.items()])
        cat_descriptions = "\n".join([f"- {k}: {v}" for k, v in self.category_mapping.items()])
        
        prompt = f"""
请对以下医疗QA数据进行分类标注：

问题：{question}
答案：{answer}

科室分类（可选择多个，最多6个）：
{dept_descriptions}

问题类别分类（可选择多个，最多6个）：
{cat_descriptions}

请返回JSON格式的分类结果：
{{
    "departments": [选择的科室编号列表],
    "categories": [选择的类别编号列表],
    "reasoning": "分类reasoning"
}}
"""
        return prompt
    
    async def _get_classification_schema(self) -> List[types.TextContent]:
        """获取分类体系说明"""
        schema_info = {
            "department_classifications": self.department_mapping,
            "category_classifications": self.category_mapping,
            "data_schema": {
                "question": {"dtype": "string", "max_length": 512},
                "answer": {"dtype": "string", "max_length": 512},
                "departments": {"dtype": "list", "max_length": 6, "description": "科室分类ID列表"},
                "categories": {"dtype": "list", "max_length": 6, "description": "问题类别分类ID列表"}
            }
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(schema_info, ensure_ascii=False, indent=2)
        )]
    
    async def _validate_annotation(self, arguments: dict) -> List[types.TextContent]:
        """验证标注结果"""
        try:
            annotated_data = arguments.get("annotated_data", [])
            validation_results = []
            
            for i, item in enumerate(annotated_data):
                errors = []
                
                # 检查必要字段
                required_fields = ["question", "answer", "departments", "categories"]
                for field in required_fields:
                    if field not in item:
                        errors.append(f"缺少字段: {field}")
                
                # 检查字符串长度
                if "question" in item and len(item["question"]) > 512:
                    errors.append("question长度超过512字符")
                if "answer" in item and len(item["answer"]) > 512:
                    errors.append("answer长度超过512字符")
                
                # 检查列表长度和内容
                if "departments" in item:
                    if not isinstance(item["departments"], list):
                        errors.append("departments必须是列表")
                    elif len(item["departments"]) > 6:
                        errors.append("departments长度不能超过6")
                    elif not all(isinstance(x, int) and 0 <= x <= 5 for x in item["departments"]):
                        errors.append("departments包含无效值（应为0-5的整数）")
                
                if "categories" in item:
                    if not isinstance(item["categories"], list):
                        errors.append("categories必须是列表")
                    elif len(item["categories"]) > 6:
                        errors.append("categories长度不能超过6")
                    elif not all(isinstance(x, int) and 0 <= x <= 7 for x in item["categories"]):
                        errors.append("categories包含无效值（应为0-7的整数）")
                
                validation_results.append({
                    "item_index": i,
                    "valid": len(errors) == 0,
                    "errors": errors
                })
            
            summary = {
                "total_items": len(annotated_data),
                "valid_items": sum(1 for r in validation_results if r["valid"]),
                "invalid_items": sum(1 for r in validation_results if not r["valid"]),
                "validation_details": validation_results
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(summary, ensure_ascii=False, indent=2)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"验证失败: {str(e)}"
            )]

async def main():
    """启动MCP服务器"""
    server_instance = QAAnnotationServer()
    
    # 使用stdio transport启动服务器
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="qa-annotation-server",
                server_version="1.0.0",
                capabilities=server_instance.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())