#!/usr/bin/env python3
"""
QA数据标注MCP客户端
连接到MCP服务器和本地ollama qwen3:32b模型，实现智能数据标注
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from typing import Dict, List, Any, Optional

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import httpx

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qa_annotation_client")

# 配置类
class Config:
    """系统配置"""
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:32b")
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("系统配置:")
        print(f"  Ollama服务器: {cls.OLLAMA_BASE_URL}")
        print(f"  使用模型: {cls.OLLAMA_MODEL}")
        print(f"  环境变量配置:")
        print(f"    export OLLAMA_BASE_URL={cls.OLLAMA_BASE_URL}")
        print(f"    export OLLAMA_MODEL={cls.OLLAMA_MODEL}")
        print()

class OllamaClient:
    """Ollama客户端"""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.model = model or Config.OLLAMA_MODEL
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """调用ollama生成响应"""
        try:
            # 构建完整的提示
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # 使用Ollama的原生API格式
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 20,
                    "num_predict": 2048
                }
            }
            
            logger.info(f"调用Ollama API: {self.base_url}/api/generate")
            logger.info(f"\n\n调用Ollama 原文: {payload}\n\n")
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            logger.info(f"Ollama响应状态: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                logger.info(f"生成文本长度: {len(generated_text)}")
                return generated_text
            else:
                logger.error(f"Ollama API错误: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"调用Ollama时出错: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return ""
    
    async def close(self):
        """关闭客户端"""
        await self.client.aclose()

class QAAnnotationClient:
    """QA数据标注客户端"""
    
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        # 使用配置系统
        self.ollama_client = OllamaClient()
        
        # 系统提示
        self.system_prompt = """
你是一个专业的医疗数据标注助手。你的任务是对医疗QA数据进行准确的分类标注。

分类规则：
1. 科室分类（departments）：分析问题和答案内容，选择最相关的科室（可多选，最多6个）
2. 问题类别分类（categories）：根据问题的性质进行分类（可多选，最多6个）

请严格按照提供的分类体系进行标注，并返回准确的JSON格式结果。
"""
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.exit_stack.aclose()
        await self.ollama_client.close()
    
    async def connect_to_server(self, server_script_path: str):
        """连接到MCP服务器"""
        try:
            server_params = StdioServerParameters(
                command="python",
                args=[server_script_path],
                env=None
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            
            await self.session.initialize()
            
            # 列出可用工具
            response = await self.session.list_tools()
            tools = response.tools
            logger.info(f"连接到MCP服务器，可用工具: {[tool.name for tool in tools]}")
            
        except Exception as e:
            logger.error(f"连接MCP服务器失败: {e}")
            raise
    
    async def get_classification_schema(self) -> Dict[str, Any]:
        """获取分类体系"""
        try:
            result = await self.session.call_tool("get_classification_schema", {})
            schema_text = result.content[0].text
            return json.loads(schema_text)
        except Exception as e:
            logger.error(f"获取分类体系失败: {e}")
            return {}
    
    async def classify_qa_item(self, question: str, answer: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """使用AI对单个QA项进行分类"""
        # 构建分类提示
        dept_info = schema.get("department_classifications", {})
        cat_info = schema.get("category_classifications", {})
        
        dept_desc = "\n".join([f"{k}: {v}" for k, v in dept_info.items()])
        cat_desc = "\n".join([f"{k}: {v}" for k, v in cat_info.items()])
        
        prompt = f"""
请对以下医疗QA数据进行精确的分类标注：

【问题】：{question}

【答案】：{answer}

【科室分类选项】：
{dept_desc}

【问题类别分类选项】：
{cat_desc}

【分类指导】：
- 仔细阅读问题和答案的医学内容
- 从科室分类中选择1-6个最相关的编号
- 从问题类别分类中选择1-6个最相关的编号
- 科室分类编号范围：0-5
- 问题类别分类编号范围：0-7

【输出格式】：
必须严格按照以下JSON格式返回，不要添加任何额外的文字说明：

{{
    "departments": [0, 3],
    "categories": [1],
    "reasoning": "这是一个关于口干症的治疗问题。涉及口腔科（3）和内科系统（0），主要是治疗方案类问题（1）"
}}

请返回JSON："""
        
        try:
            response = await self.ollama_client.generate(prompt, self.system_prompt)
            
            if not response.strip():
                raise ValueError("模型返回空响应")
            
            logger.info(f"模型原始响应: {response[:200]}...")
            
            # 尝试解析JSON响应
            response = response.strip()
            
            # 尝试提取JSON部分
            json_start = -1
            json_end = -1
            
            # 查找第一个 {
            for i, char in enumerate(response):
                if char == '{':
                    json_start = i
                    break
            
            # 查找最后一个 }
            for i in range(len(response) - 1, -1, -1):
                if response[i] == '}':
                    json_end = i + 1
                    break
            
            if json_start != -1 and json_end != -1:
                json_text = response[json_start:json_end]
            else:
                # 如果找不到{}，尝试整个响应
                json_text = response
            
            logger.info(f"提取的JSON文本: {json_text}")
            
            classification_result = json.loads(json_text)
            
            # 验证结果格式
            if "departments" not in classification_result or "categories" not in classification_result:
                raise ValueError("分类结果缺少必要字段")
            
            # 确保是列表且在有效范围内
            departments = []
            categories = []
            
            # 处理departments
            if isinstance(classification_result["departments"], list):
                for d in classification_result["departments"]:
                    try:
                        d_int = int(d)
                        if 0 <= d_int <= 5:
                            departments.append(d_int)
                    except (ValueError, TypeError):
                        continue
            
            # 处理categories  
            if isinstance(classification_result["categories"], list):
                for c in classification_result["categories"]:
                    try:
                        c_int = int(c)
                        if 0 <= c_int <= 7:
                            categories.append(c_int)
                    except (ValueError, TypeError):
                        continue
            
            # 限制数量
            departments = departments[:6]
            categories = categories[:6]
            
            # 确保至少有一个分类
            if not departments:
                departments = [0]
            if not categories:
                categories = [0]
            
            return {
                "departments": departments,
                "categories": categories,
                "reasoning": classification_result.get("reasoning", "")
            }
            
        except Exception as e:
            logger.error(f"AI分类失败: {e}")
            logger.error(f"模型响应: {response if 'response' in locals() else 'No response'}")
            # 返回默认分类
            return {
                "departments": [0],
                "categories": [0],
                "reasoning": f"分类失败，使用默认值。错误：{str(e)}"
            }
    
    async def annotate_qa_data(self, qa_data: Dict[str, Any]) -> Dict[str, Any]:
        """标注QA数据"""
        try:
            # 获取分类体系
            schema = await self.get_classification_schema()
            if not schema:
                raise ValueError("无法获取分类体系")
            
            questions = qa_data.get("questions", [])
            answers = qa_data.get("answers", [])
            
            if not questions or not answers:
                raise ValueError("缺少问题或答案数据")
            
            logger.info(f"开始标注 {len(questions)} 个QA项...")
            
            annotated_items = []
            
            for i, (question_group, answer) in enumerate(zip(questions, answers)):
                # 处理问题格式（取第一个问题）
                main_question = question_group[0] if isinstance(question_group, list) else str(question_group)
                
                logger.info(f"正在标注第 {i+1}/{len(questions)} 项...")
                
                # 使用AI进行分类
                classification = await self.classify_qa_item(main_question, answer, schema)
                
                # 构建标注项
                annotated_item = {
                    "question": main_question[:512],  # 长度限制
                    "answer": answer[:512],           # 长度限制
                    "departments": classification["departments"],
                    "categories": classification["categories"],
                    "reasoning": classification["reasoning"]
                }
                
                annotated_items.append(annotated_item)
                
                # 显示进度
                logger.info(f"完成第 {i+1} 项，科室: {classification['departments']}, 类别: {classification['categories']}")
            
            # 验证标注结果
            validation_result = await self.session.call_tool("validate_annotation", {
                "annotated_data": annotated_items
            })
            
            validation_info = json.loads(validation_result.content[0].text)
            
            result = {
                "annotated_data": annotated_items,
                "total_items": len(annotated_items),
                "validation": validation_info,
                "schema": schema.get("data_schema", {})
            }
            
            logger.info(f"标注完成！总计 {len(annotated_items)} 项，验证通过 {validation_info['valid_items']} 项")
            
            return result
            
        except Exception as e:
            logger.error(f"标注过程出错: {e}")
            raise

async def main():
    """主函数"""
    # 示例数据
    sample_qa_data = {
        "questions": [["口干的治疗方案是什么?", "请描述口干的治疗方案"]],
        "answers": ["口干症的治疗包括病因治疗和对症治疗。对因治疗在明确病因的情况下是最有效的，如药物性口干，通过调整药物及其剂量，可缓解口干。对唾液消耗增加而产生的口干，可通过消除张口呼吸等原因来解决。如果是由于唾液腺实质破坏所引起的口感，如头颈部恶性肿瘤放疗后、舍格伦综合征，目前主要通过对症治疗来缓解口干，减少并发症。"]
    }
    
    # 获取服务器脚本路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(current_dir, "qa_annotation_server.py")
    
    if not os.path.exists(server_script):
        logger.error(f"找不到服务器脚本: {server_script}")
        logger.info("请确保 qa_annotation_server.py 文件在同一目录下")
        return
    
    async with QAAnnotationClient() as client:
        try:
            logger.info("连接到MCP服务器...")
            await client.connect_to_server(server_script)
            
            logger.info("开始数据标注...")
            result = await client.annotate_qa_data(sample_qa_data)
            
            # 保存结果
            output_file = "annotated_qa_data.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"标注结果已保存到: {output_file}")
            
            # 显示结果摘要
            print("\n" + "="*50)
            print("标注结果摘要")
            print("="*50)
            print(f"总计标注项目: {result['total_items']}")
            print(f"验证通过项目: {result['validation']['valid_items']}")
            print(f"验证失败项目: {result['validation']['invalid_items']}")
            print("\n标注样例:")
            
            for i, item in enumerate(result['annotated_data'][:3], 1):  # 显示前3个
                print(f"\n项目 {i}:")
                print(f"问题: {item['question'][:100]}...")
                print(f"科室分类: {item['departments']}")
                print(f"问题类别: {item['categories']}")
                print(f"分类reasoning: {item['reasoning'][:200]}...")
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            sys.exit(1)

if __name__ == "__main__":
    # 打印配置信息
    Config.print_config()
    
    # 检查ollama是否运行
    try:
        import httpx
        client = httpx.Client()
        response = client.get(f"{Config.OLLAMA_BASE_URL}/api/tags")
        if response.status_code != 200:
            print(f"错误: 无法连接到Ollama服务器 ({Config.OLLAMA_BASE_URL})")
            print("请确保Ollama正在运行: ollama serve")
            sys.exit(1)
        
        # 检查模型是否存在
        models = response.json().get("models", [])
        model_names = [model["name"] for model in models]
        if Config.OLLAMA_MODEL not in model_names:
            print(f"错误: 未找到{Config.OLLAMA_MODEL}模型")
            print(f"请先下载模型: ollama pull {Config.OLLAMA_MODEL}")
            print(f"当前可用模型: {model_names}")
            sys.exit(1)
        
        print("✓ Ollama服务器运行正常")
        print(f"✓ {Config.OLLAMA_MODEL}模型可用")
        print(f"✓ 使用服务器地址: {Config.OLLAMA_BASE_URL}")
        print()
        
    except Exception as e:
        print(f"错误: 无法检查Ollama状态: {e}")
        print(f"请确保Ollama已安装并运行在: {Config.OLLAMA_BASE_URL}")
        sys.exit(1)
    
    asyncio.run(main())