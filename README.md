# Medical RAG 医疗问答智能体

基于 Milvus + LangChain 搭建的医疗问答智能体，采用RAG技术，提供准确和安全的医疗建议。

## 主要特性

### 自动化构建高质量数据
- **自动化标注流水线**：支持HTTP、本地GPU推理，可配置推理过程参数，加速标注流程
- **自动化管理词表**：采用多线程+领域分词器，自动化构建和管理词表，便于后续入库，增加查询准确性

### 混合检索架构
- **稠密向量检索**：支持 Ollama、OpenAI、HuggingFace 等嵌入模型，加速http嵌入以及使用本地批量嵌入
- **稀疏向量检索**：本项目可自动管理词表，使用了基于医疗领域分词优化的 BM25 算法
- **混合重排**：RRF 或加权融合多路检索结果

### 医疗领域优化
- **专业分类体系**：6大科室分类 + 8大问题类别
- **智能标注**：支持多种LLM后端的自动数据标注
- **医疗分词**：使用 pkuseg 医疗领域分词模型

### 高性能数据库设计
- **向量数据库**：基于 Milvus v2.6+ 高性能向量检索
- **并发处理**：支持批量嵌入和并发查询
- **灵活配置**：YAML 配置文件支持多环境部署，查询数据只需要编写yaml文件即可

### 高效率接口设计
- **封装Milvus高频接口**：便于工具调用，以及yaml文件的查询实现
- **prompt管理**：可配置多版本prompt，只需要对应修改yaml中的配置即可

## 🚀 快速开始

### 环境准备

```bash
# 1. 克隆项目
git clone https://github.com/yolo-hyl/medical-rag
cd medical-rag/src

# 2. 安装项目
pip install -e .

# 3. 启动 Milvus (使用 Docker)
cd Milvus
bash standalone_embed.sh start

# 4. 启动 Ollama (可选)
ollama serve
ollama pull bge-m3:latest  # 嵌入模型（可自定义）
ollama pull qwen3:32b        # 标注模型（可自定义）
```

### 基础配置

编辑 `src/MedicalRag/config/default.yaml` 配置文件，部分示例如下，详见配置文件：

```yaml
milvus:
  client:
    uri: "http://localhost:19530"
    token: "root:Milvus"
  collection:
    name: "qa_knowledge"

embedding:
  dense:
    provider: ollama
    model: "bge-m3:latest"
    base_url: "http://localhost:11434"
```

## 使用流程

### 数据标注

对原始QA数据进行智能标注，自动分类科室和问题类别：

```bash
# 配置标注参数
vim src/MedicalRag/config/data/annotator.yaml

# 运行标注
python scripts/annotation.py src/MedicalRag/config/data/annotator.yaml
```

**标注功能**：
- 支持医疗科室分类（内科、外科、妇产儿科等6大类）
- 支持问题类别分类（诊断症状、治疗方案等8大类）
- 多LLM后端支持（Ollama、vLLM、OpenAI等）
- 断点续标和批量处理

### 构建词表

为BM25稀疏向量构建医疗领域词汇表：

```bash
python scripts/build_vocab.py
```

**词表特性**：
- 基于医疗领域语料训练
- 支持并行分词处理
- 生成 `vocab.pkl.gz` 词表文件，可指定目录或由项目进行自动管理

### 创建Milvus集合

在Milvus中创建向量数据库集合：

#### 编辑配置文件
```code yaml
milvus:
  client:
  # 基础配置信息
  collection：
  # 集合配置信息
  schema:
  # 字段信息
  index:
  # 索引
  search:
  # 基础查询定义
  write:
  # 插入配置设定
embedding:
  dense:
  # 密集向量生成定义
  sparse_bm25:
  # 稀疏向量定义
```
#### 自动化创建

```bash
# 创建集合
python scripts/create_collection.py -c src/MedicalRag/config/default.yaml

# 强制重建集合
python scripts/create_collection.py --force-recreate

# 测试连接是否可用
python scripts/create_collection.py --connection-only
```

**集合特性**：
- 支持稠密向量 + 稀疏向量混合存储
- 自动创建 HNSW 和倒排索引
- 支持分区键优化查询性能

### 数据入库

将处理好的数据导入向量数据库：

```bash
python scripts/insert_data_to_collection.py
```

**入库流程**：
- 自动向量化（稠密 + 稀疏）
- 批量插入优化
- 支持增量更新

### 查询检索

执行混合检索查询：

#### 使用默认配置查询
在定义default配置时，会定义search，这里可以配置查询规则

#### 使用自定义配置查询
也可以使用自己配置的yaml或者字典进行查询，具体可以查看config/search下的yaml文件

#### 开始使用
```bash
# 使用搜索配置
python scripts/search_pipline.py --search-config src/MedicalRag/config/search/search_answer.yaml

# 测试配置字典自定义创建查询
python scripts/search_pipline.py --test-config-dict
```

**检索特性**：
- 单次查询和批量查询
- 支持过滤条件和分页
- 动态调整检索通道权重

## 核心工具类

### RAG搜索工具

```python
from MedicalRag.tools.rag_search_tool import RAGSearchTool

# 从配置文件创建
tool = RAGSearchTool("config/search.yaml")

if tool.is_ready():
    # 单个查询
    results = tool.search("梅毒的症状有哪些？")
    
    # 批量查询
    results = tool.search(["梅毒", "高血压"])
    
    # 带过滤条件
    results = tool.search("梅毒", filters={"dept_pk": "3"})
```

## 项目结构

```
medical-rag/
├── src/MedicalRag/           # 核心源码
│   ├── core/                 # 核心组件
│   │   ├── llm/             # LLM客户端
│   │   ├── vectorstore/     # Milvus操作
│   │   └── embeddings/      # 嵌入模型
│   ├── data/                # 数据处理
│   │   ├── loader/          # 数据加载器
│   │   ├── processor/       # 数据处理器
│   │   └── annotator/       # 数据标注器
│   ├── pipeline/            # 流水线
│   │   ├── ingestion/       # 数据入库
│   │   └── query/           # 查询检索
│   ├── config/              # 配置文件
│   └── tools/               # 工具类
├── scripts/                 # 主要功能脚本
├── Milvus/                  # Milvus部署脚本
```

## 配置说明

### 主要配置文件

- `src/MedicalRag/config/default.yaml` - 主配置文件
- `src/MedicalRag/config/search/search_answer.yaml` - 搜索专用配置
- `src/MedicalRag/config/data/annotator.yaml` - 标注配置

### 关键配置项

```yaml
# Milvus配置
milvus:
  client:
    uri: "http://localhost:19530"
  collection:
    name: "qa_knowledge"

# 嵌入模型配置
embedding:
  dense:
    provider: ollama
    model: "bge-m3:latest"
  sparse_bm25:
    vocab_path: "vocab.pkl.gz"

# 搜索通道配置
search:
  channels:
    - name: sparse_q          # 查询稀疏向量
      weight: 0.3
    - name: sparse_doc        # 文档稀疏向量  
      weight: 0.4
    - name: dense_doc         # 文档稠密向量
      weight: 0.3
```

## 高级功能

### 自定义LLM后端

支持多种LLM后端进行数据标注：

```python
# Ollama
llm_config = {
    "type": "ollama",
    "model_name": "qwen3:32b",
    "ollama": {"base_url": "http://localhost:11434"}
}

# vLLM
llm_config = {
    "type": "vllm", 
    "model_name": "Qwen/Qwen2-7B-Instruct",
    "vllm": {"base_url": "http://localhost:8000"}
}

# OpenAI
llm_config = {
    "type": "openai",
    "openai": {"api_key": "your-key"}
}
```

### 动态检索配置

运行时动态调整检索策略：

```python
# 更新通道权重
tool.update_search_config({
    "channels": [
        {"name": "sparse_q", "weight": 0.5},
        {"name": "dense_doc", "weight": 0.5}
    ]
})
```

## 性能优化

- **并发处理**：支持批量嵌入和并发查询
- **索引优化**：HNSW索引 + 稀疏倒排索引
- **缓存机制**：嵌入结果缓存和词表缓存
- **分区策略**：按科室分区提升查询性能

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 许可证

MIT License

---

## 常见问题

**Q: 如何切换嵌入模型？**
A: 修改配置文件中的 `embedding.dense.provider` 和 `model` 字段。

**Q: 如何自定义医疗分类体系？**  
A: 修改 `src/MedicalRag/config/prompts.py` 中的分类定义。

**Q: 检索结果不理想怎么办？**
A: 可以调整 `search.channels` 中各通道的权重，或重新训练词表。