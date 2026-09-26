# GraphRAG-Bench Examples

This directory contains example implementations for running inference on the GraphRAG-Bench dataset using various RAG frameworks. Each framework has a dedicated implementation file that generates prediction outputs compatible with our unified evaluation pipeline.

## 🛠 Installation Guide

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

## 🚀 Running Example
Next, we provide detailed instructions on how to use GraphRAG-Bench to evaluate each framework. Specifically, we introduce how to perform index construction and batch inference for each framework. Note that the evaluation code is standardized across all frameworks to ensure fair comparison.
### 1. Indexing and inference
#### a. LightRAG
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
#### b. fast-graphrag
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

#### c. hipporag2

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
### 2. Evaluation
#### a. Generation
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

#### b. Retrieval
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


