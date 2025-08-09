# Medical Agent
使用 Milvus + Langchain 搭建的医疗问答智能体，采用RAG技术，向用户提供更加准确和安全的事实医疗建议

## 初始化数据 Pipeline


  - 元数据用 Pydantic + Milvus JSON 字段/动态字段存储可演化结构；

  - LLM 标注用 结构化输出（PydanticOutputParser / with_structured_output()），显著降低脏 JSON。

  - 向量嵌入通过配置切换 OpenAI/HF；注意新版包名与弃用路径。

  - Milvus 入库：langchain-milvus 集成 + pymilvus 细粒度控制；分区优先 Partition Key；索引首选 HNSW、大规模用 IVF_PQ。

## 新增数据 Pipeline

  - 改动检测（哈希/主键），优先使用 Upsert，避免重复与“脏影子”。

  - 规则查询

  - 统一的 filter_builder.py 产出 Milvus 布尔表达式与 JSONPath 过滤；支持模板化与批量参数替换。

## 过时数据定位与删除

  - 集合级 TTL（全部过期型场景）+ 记录 last_seen_at/version 实施精确清理；用 Delete/Upsert 执行。

## 评估 Pipeline

  - 检索侧：Recall@k/MRR/nDCG；

  - 生成侧：RAGAS faithfulness/answer relevancy/context 相关指标；LangSmith 自带 RAG 测评流水线（数据集→运行→打分）。

## 智能体

  - 用 LangGraph 组织 Agent（状态、工具循环、人工批准）；模型侧用 .bind_tools() 声明工具与入参；入口先经 Router 判断意图再走工具/检索流。

## 配置化

  - 使用 pydantic-settings 管理所有可变项（BaseURL、API Key、模型名、嵌入维度、索引参数、检索策略、Prompt 文案等），V2 起 BaseSettings 移至独立包。亦可选 Hydra 做多环境/多组合切换。