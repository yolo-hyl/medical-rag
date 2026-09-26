---
configs:
- config_name: novel
  data_files: "Datasets/Questions/novel_questions.json"
- config_name: medical
  data_files: "Datasets/Questions/medical_questions.json"
---

<div align="center">
  
 # GraphRAG-Bench : A Comprehensive Benchmark for Evaluating Graph Retrieval-Augmented Generation Models

[![Static Badge](https://img.shields.io/badge/arxiv-2506.05690-ff0000?style=for-the-badge&labelColor=000)](https://arxiv.org/abs/2506.05690)  [![Static Badge](https://img.shields.io/badge/huggingface-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench)  [![Static Badge](https://img.shields.io/badge/leaderboard-steelblue?style=for-the-badge&logo=googlechrome&logoColor=ffffff)](https://graphrag-bench.github.io/)  [![Static Badge](https://img.shields.io/badge/license-mit-teal?style=for-the-badge&labelColor=000)](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/LICENSE)

  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉News</a> •
    <a href="#about" style="text-decoration: none; font-weight: bold;">📖About</a> •
    <a href="#leaderboards" style="text-decoration: none; font-weight: bold;">🏆Leaderboards</a> •
    <a href="#task-examples" style="text-decoration: none; font-weight: bold;">🧩Task Examples</a> 
    
  </p>
  <p>
  <a href="#getting-started" style="text-decoration: none; font-weight: bold;">🔧Getting Started</a> •
    <a href="#contribution--contact" style="text-decoration: none; font-weight: bold;">📬Contact</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">📝Citation</a> 
  </p>
</div>

This repository is for the GraphRAG-Bench project, a comprehensive benchmark for evaluating Graph Retrieval-Augmented Generation models.
![pipeline](./pipeline.jpg)

<h2 id="news">🎉 News</h2>

- **[2025-05-25]** We release [GraphRAG-Bench](https://graphrag-bench.github.io), the benchmark for evaluating GraphRAG models.
- **[2025-05-14]** We release the [GraphRAG-Bench dataset](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench).
- **[2025-01-21]** We release the [GraphRAG survey](https://github.com/DEEP-PolyU/Awesome-GraphRAG).

<h2 id="about">📖 About</h2>

- Introduces Graph Retrieval-Augmented Generation (GraphRAG) concept
- Compares traditional RAG vs GraphRAG approach
- Explains research objective: Identify scenarios where GraphRAG outperforms traditional RAG
- Visual comparison diagram of RAG vs GraphRAG

![overview](./RAGvsGraphRAG.jpg)

<details>
<summary>
  More Details
</summary>
Graph retrieval-augmented generation (GraphRAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) with external knowledge. It leverages graphs to model the hierarchical structure between specific concepts, enabling more coherent and effective knowledge retrieval for accurate reasoning. Despite its conceptual promise, recent studies report that GraphRAG frequently underperforms vanilla RAG on many real-world tasks. This raises a critical question: Is GraphRAG really effective, and in which scenarios do graph structures provide measurable benefits for RAG systems? To address this, we propose GraphRAG-Bench, a comprehensive benchmark designed to evaluate GraphRAG models on both hierarchical knowledge retrieval and deep contextual reasoning. GraphRAG-Bench features a comprehensive dataset with tasks of increasing difficulty, covering fact retrieval, complex reasoning, contextual summarization, and creative generation, and a systematic evaluation across the entire pipeline, from graph construction and knowledge retrieval to final generation. Leveraging this novel benchmark, we systematically investigate the conditions when GraphRAG surpasses traditional RAG and the underlying reasons for its success, offering guidelines for its practical application.
</details>

<h2 id="leaderboards">🏆 Leaderboards</h2>

Two domain-specific leaderboards with comprehensive metrics:

**1. GraphRAG-Bench (Novel)**
- Evaluates models on literary/fictional content

**2. GraphRAG-Bench (Medical)**
- Evaluates models on medical/healthcare content

**Evaluation Dimensions:**
- Fact Retrieval (Accuracy, ROUGE-L)
- Complex Reasoning (Accuracy, ROUGE-L)
- Contextual Summarization (Accuracy, Coverage)
- Creative Generation (Accuracy, Factual Score, Coverage)

<h2 id="task-examples">🧩 Task Examples</h2>
Four difficulty levels with representative examples:

**Level 1: Fact Retrieval**  
*Example: "Which region of France is Mont St. Michel located?"*

**Level 2: Complex Reasoning**  
*Example: "How did Hinze's agreement with Felicia relate to the perception of England's rulers?"*

**Level 3: Contextual Summarization**  
*Example: "What role does John Curgenven play as a Cornish boatman for visitors exploring this region?"*

**Level 4: Creative Generation**  
*Example: "Retell King Arthur's comparison to John Curgenven as a newspaper article."*


<h2 id="getting-started">🔧 Getting Started(GraphRAG-Bench Examples)</h2>

The  **'Examples'** directory contains example implementations for running inference on the GraphRAG-Bench dataset using various RAG frameworks. Each framework has a dedicated implementation file that generates prediction outputs compatible with our unified evaluation pipeline.

### Installation Guide

**To prevent dependency conflicts, we strongly recommend using separate Conda environments for each framework:**

We use the installation of LightRAG as an example. For other frameworks, please refer to their respective installation instructions.
```bash
# Create and activate environment (example for LightRAG)
conda create -n lightrag python=3.10 -y
conda activate lightrag

# Install LightRAG
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
pip install -e .

```

### Running Example
Next, we provide detailed instructions on how to use GraphRAG-Bench to evaluate each framework. Specifically, we introduce how to perform index construction and batch inference for each framework. Note that the evaluation code is standardized across all frameworks to ensure fair comparison.
#### 1. Indexing and inference
##### a. LightRAG
Before running the above script, you need to modify the source code(LightRAG) to enable extraction of the corresponding context used during generation. Please make the following changes:
1. In lightrag/operate.py, update the kg_query method to return the context along with the response:
```python
# Original Code
async def kg_query(...) -> str | AsyncIterator[str]:
  return response

# Modified Code
async def kg_query(...) -> tuple[str, str] | tuple[AsyncIterator[str], str]:
  return response, context
```
2. In lightrag/lightrag.py, update the aquery method to receive and return the context when calling kg_query:
```python
# Modified Code
async def aquery(...):
  ...
  if param.mode in ["local", "global", "hybrid"]:
      response, context = await kg_query(...)
  ...
  return response, context

```
Then you can run the following command to indexing and inference:
```shell
export LLM_API_KEY=your_actual_api_key_here

python run_lightrag.py \
  --subset medical \
  --base_dir ./Examples/lightrag_workspace \
  --model_name bge-large-en-v1.5 \
  --embed_model bge-base-en \
  --retrieve_topk 5 \
#   --sample 100 \
  --llm_base_url https://api.openai.com/v1

```
##### b. fast-graphrag
```shell
export LLM_API_KEY=your_actual_api_key_here

python run_fast-graphrag.py \
  --subset medical \
  --base_dir ./Examples/fast-graphrag_workspace \
  --model_name gpt-4o-mini \
  --embed_model_path bge-large-en-v1.5 \
#   --sample 100 \
  --llm_base_url https://api.openai.com/v1

```

##### c. hipporag2

```shell
export OPENAI_API_KEY=your_actual_api_key_here

python run_hipporag2.py \
  --subset medical \
  --base_dir ./Examples/hipporag2_workspace \
  --model_name gpt-4o-mini \
  --embed_model_path contriever \
#   --sample 100 \
  --llm_base_url https://api.openai.com/v1
```
We will continue updating other GraphRAG frameworks as much as possible. If you wish to integrate a different framework, you can refer to the structure of our result format. As long as your returned output matches the following fields, the evaluation code will run successfully:
```json
{
  "id": q["id"],
  "question": q["question"],
  "source": corpus_name,
  "context": context,
  "evidence": q["evidence"],
  "question_type": q["question_type"],
  "generated_answer": predicted_answer,
  "gold_answer": q["answer"]
}

```
#### 2. Evaluation
##### a. Generation
```shell
cd Evaluation
export OPENAI_API_KEY=your_actual_api_key_here

python -m Evaluation.generation_eval \
  --model gpt-4-turbo \
  --base_url https://api.openai.com/v1 \
  --bge_model BAAI/bge-large-en-v1.5 \
  --data_file ./results/lightrag.json \
  --output_file ./results/evaluation_results.json
```

##### b. Retrieval
```shell
cd Evaluation
export OPENAI_API_KEY=your_actual_api_key_here

python -m Evaluation.retrieval_eval \
  --model gpt-4-turbo \
  --base_url https://api.openai.com/v1 \
  --bge_model BAAI/bge-large-en-v1.5 \
  --data_file ./results/lightrag.json \
  --output_file ./results/evaluation_results.json
```

## <h2 id="contribution--contact">📬 Contribution & Contact</h2>

Contributions to improve the benchmark website are welcome. Please contact the project team via <a href="mailto:GraphRAG@hotmail.com">GraphRAG@hotmail.com</a>.

## <h2 id="citation">📝 Citation</h2>

If you find this benchmark helpful, please cite our paper:
```
@article{xiang2025use,
  title={When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation},
  author={Xiang, Zhishang and Wu, Chuanjie and Zhang, Qinggang and Chen, Shengyuan and Hong, Zijin and Huang, Xiao and Su, Jinsong},
  journal={arXiv preprint arXiv:2506.05690},
  year={2025}
}
```

