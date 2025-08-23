# Medical RAG v2 使用指南

基于 LangChain-Milvus 的简化医疗RAG系统，去除冗余，专注核心功能。

## 🏗️ 项目架构

```
medical-rag-v2/
├── src/medical_rag/
│   ├── config/              # 配置系统
│   │   ├── models.py        # 配置数据模型
│   │   └── loader.py        # 配置加载器
│   ├── core/                # 核心组件
│   │   └── components.py    # LLM/嵌入/向量存储创建
│   ├── knowledge/           # 知识库功能
│   │   ├── bm25.py         # BM25处理（保留原实现）
│   │   ├── ingestion.py    # 数据入库
│   │   └── annotation.py   # 自动标注
│   ├── rag/                # RAG功能
│   │   └── basic_rag.py    # 基础RAG和智能体RAG
│   └── prompts/            # Prompt管理
│       └── templates.py    # Prompt模板
├── config/                 # 配置文件
└── scripts/               # 使用脚本
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install langchain langchain-milvus langchain-openai langchain-ollama
pip install langchain-community datasets pymilvus pkuseg

# 启动 Milvus
docker run -p 19530:19530 milvusdb/milvus:latest

# 启动 Ollama (如果使用Ollama)
ollama serve
ollama pull bge-m3:latest
ollama pull qwen3:32b
```

### 2. 配置设置

编辑 `config/app_config.yaml`:

```yaml
milvus:
  uri: "http://localhost:19530"
  collection_name: "medical_knowledge"

embedding:
  dense:
    provider: "ollama"  # 或 "openai"
    model: "bge-m3:latest"
    base_url: "http://localhost:11434"
  sparse:
    manager: "self"  # 或 "milvus" 使用内置BM25

llm:
  provider: "ollama"  # 或 "openai"  
  model: "qwen3:32b"
  base_url: "http://localhost:11434"

data:
  path: "/path/to/your/data.jsonl"
  question_field: "question"
  answer_field: "answer"
```

## 📚 核心功能使用

### 功能1: 构建BM25词表（自管理BM25）

```bash
python scripts/01_build_vocab.py
```

**何时使用**: 当配置中 `sparse.manager: "self"` 时需要

### 功能2: 数据入库

```bash
python scripts/02_ingest_data.py
```

**功能说明**:
- 自动加载数据集（支持JSON/JSONL/Parquet）
- 根据配置选择BM25方案（自管理 vs Milvus内置）
- 批量向量化和入库
- 使用 langchain-milvus 的标准接口

### 功能3: 自动标注

```bash
python scripts/03_annotate_data.py \
  --input data/raw_qa.jsonl \
  --output data/annotated_qa.json \
  --question-field question \
  --answer-field answer
```

**功能说明**:
- 医疗QA自动分类标注
- 6大科室 + 8大问题类别
- 基于LangChain的LLM调用
- 支持批量处理和错误重试

### 功能4: 基础RAG

```bash
python scripts/04_basic_rag.py
```

**功能说明**:
- 使用 langchain-milvus 的混合检索
- 基于检索结果生成回答
- 硬编码的RAG流程
- 支持过滤条件

**交互示例**:
```
请输入您的问题: 高血压的症状有哪些？

回答: 高血压的主要症状包括头痛、头晕、心悸...

参考资料 (3 条):
1. medical_source
   高血压是一种常见的心血管疾病...
```

### 功能5: 智能体RAG

```bash
python scripts/05_agent_rag.py
```

**功能说明**:
- 自主确定检索参数和内容
- 知识库检索 + 网络搜索结合
- 多源信息综合
- 简单工具调用（计算器等）

**智能特性**:
1. **自主检索**: 分析问题自动确定搜索策略
2. **多源信息**: 知识库为主，网络搜索补充  
3. **信息综合**: LLM综合多个信息源
4. **工具调用**: 支持计算器、网络搜索等工具

## 🔧 配置选项详解

### Milvus内置BM25 vs 自管理BM25

#### 使用Milvus内置BM25（推荐）
```yaml
embedding:
  sparse:
    manager: "milvus"  # Milvus 2.5+支持
```

**优势**:
- 无需构建词表
- 自动处理BM25计算
- 更简洁的架构

#### 使用自管理BM25（保留原实现）
```yaml
embedding:
  sparse:
    manager: "self"
    vocab_path: "vocab.pkl.gz"
    domain_model: "medicine"  # 医疗领域分词
```

**优势**:
- 保持原项目的高性能实现
- 自定义BM25参数
- 不受Milvus版本限制

### LLM提供商配置

#### Ollama（内网部署）
```yaml
llm:
  provider: "ollama"
  model: "qwen3:32b"
  base_url: "http://localhost:11434"
```

#### OpenAI（支持代理）
```yaml
llm:
  provider: "openai"  
  model: "gpt-4o-mini"
  api_key: "your-key"
  base_url: "https://api.openai.com/v1"
  proxy: "http://localhost:10809"  # 可选代理
```

## 🛠️ 高级用法

### 程序化使用

```python
from medical_rag.config.loader import load_config_from_file
from medical_rag.rag.basic_rag import create_basic_rag, create_agent_rag

# 加载配置
config = load_config_from_file("config/app_config.yaml")

# 基础RAG
basic_rag = create_basic_rag(config)
answer = basic_rag.answer("糖尿病的治疗方法？")

# 智能体RAG
agent_rag = create_agent_rag(config, enable_web_search=True)
detailed_result = agent_rag.answer("最新的癌症治疗进展", return_details=True)
```

### 自定义Prompt

```python
from medical_rag.prompts.templates import register_prompt_template

# 注册自定义模板
register_prompt_template("custom_medical", {
    "system": "你是专业医生...",
    "user": "患者问题: {input}\n请提供专业建议"
})
```

### 混合检索配置

```python
from medical_rag.core.components import KnowledgeBase

kb = KnowledgeBase(config)

# 带过滤的检索
results = kb.search(
    query="高血压治疗", 
    k=10,
    filter={"source": "权威指南"}
)

# 转换为检索器
retriever = kb.as_retriever(search_kwargs={"k": 5})
```

## 📊 性能对比

| 特性 | 原项目 | 重构后 |
|------|--------|--------|
| 代码行数 | ~3000+ | ~1500 |
| 配置复杂度 | 高 | 简化 |
| Milvus操作 | 自实现 | langchain-milvus |
| BM25支持 | 仅自管理 | 双重支持 |
| LLM集成 | 自实现 | langchain标准 |
| RAG链 | 手工组装 | langchain LCEL |

## 🔍 故障排除

### 常见问题

1. **Milvus连接失败**
```bash
# 检查Milvus服务
docker ps | grep milvus
curl http://localhost:19530/health
```

2. **词表构建失败**
```python
# 检查pkuseg医疗模型
import pkuseg
seg = pkuseg.pkuseg(model_name="medicine")
```

3. **向量维度不匹配**
```yaml
# 确保配置的维度与模型一致
embedding:
  dense:
    dimension: 1024  # 需要与实际嵌入模型维度匹配
```

### 性能调优

```yaml
# 批量大小调优
data:
  batch_size: 50  # 根据内存调整

# 检索参数调优  
search:
  top_k: 10      # 检索数量
  rrf_k: 100     # RRF重排参数
```

## 📖 与原项目对比

### 保留的优势
✅ 高性能BM25实现（医疗分词 + 停用词过滤）  
✅ Prompt管理方式  
✅ 医疗领域分类体系  
✅ 自动标注功能  

### 重构的改进
🚀 使用langchain-milvus替代自实现  
🚀 支持Milvus内置BM25  
🚀 简化配置系统  
🚀 使用langchain标准RAG组件  
🚀 代码量减少50%+  

### 新增功能
🆕 智能体RAG（自主检索+多源信息）  
🆕 双重BM25支持（自管理+Milvus内置）  
🆕 更灵活的LLM配置（支持代理）  
🆕 标准langchain接口兼容性  

---

这个重构版本在保持原项目核心优势的同时，充分利用了langchain生态的成熟组件，大大简化了架构和维护成本。