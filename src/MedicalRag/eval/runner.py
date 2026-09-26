"""
统一评测流程

不同的库（Milvus 混合检索 RAG、图谱 RAG、第三方实现等）走完全相同的流程：
同一批样本、同一套指标、同一个评测模型，分数才可以横向比较。

评测只追求一致、可复现，不做并发：

    samples = load_samples("data/eval/new_qa_200.jsonl", "new_question", "answer", sample_size=10)
    samples = asyncio.run(collect(samples, answer_fn))   # 生成答案
    save_samples(samples, "out/medical_rag.jsonl")       # 落盘，之后可重复打分
    print(score(samples, eval_llm, eval_embeddings))     # 打分

本模块只依赖 ragas / datasets / LangChain 基础类型，不 import MedicalRag 的其他模块。
"""
from __future__ import annotations

import inspect
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Sequence, Tuple, Union

import pandas as pd
from datasets import load_dataset
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 统一答题函数：输入问题，返回 (答案, 上下文列表)，同步、异步实现都可以
AnswerFn = Callable[[str], Union[Tuple[str, List[str]], Awaitable[Tuple[str, List[str]]]]]


@dataclass
class EvalSample:
    """一条评测样本，也是 JSONL 每一行的格式"""
    question: str
    reference: str                                      # 参考答案
    answer: str = ""                                    # 被测系统的回答
    contexts: List[str] = field(default_factory=list)   # 被测系统检索到的上下文
    error: str = ""                                     # 生成答案时的错误信息


def load_samples(
    path: str,
    question_field: str,
    reference_field: str,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> List[EvalSample]:
    """读取数据集并采样。固定 seed，保证每个库拿到完全相同的样本。"""
    dataset = load_dataset("json", data_files=path, split="train")
    indices = list(range(len(dataset)))
    if sample_size is not None and sample_size < len(indices):
        random.Random(seed).shuffle(indices)
        indices = sorted(indices[:sample_size])

    return [
        EvalSample(
            question=str(dataset[i][question_field]),
            reference=str(dataset[i][reference_field]),
        )
        for i in indices
    ]


async def collect(samples: Sequence[EvalSample], answer_fn: AnswerFn) -> List[EvalSample]:
    """逐条串行调用 answer_fn，填入 answer / contexts；单条失败记录错误信息后继续。"""
    for sample in tqdm(samples, desc="生成答案"):
        try:
            result = answer_fn(sample.question)
            if inspect.isawaitable(result):
                result = await result
            sample.answer, sample.contexts = result
            sample.error = ""
        except Exception as e:
            logger.exception(f"问题生成答案失败：{sample.question}")
            sample.answer, sample.contexts, sample.error = "", [], str(e)
    return list(samples)


def save_samples(samples: Sequence[EvalSample], path: str) -> None:
    """写 JSONL：答案落盘后，可以在不重新生成的情况下重复打分、对比不同的库"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
    logger.info(f"已保存 {len(samples)} 条样本到 {out}")


def load_saved(path: str) -> List[EvalSample]:
    """读回 save_samples 写出的 JSONL"""
    with open(path, "r", encoding="utf-8") as f:
        return [EvalSample(**json.loads(line)) for line in f if line.strip()]


def score(
    samples: Sequence[EvalSample],
    llm: BaseChatModel,
    embeddings: Embeddings,
    metrics: Optional[list] = None,
) -> EvaluationResult:
    """用 ragas 打分。

    同步函数，需要在事件循环之外调用（先 asyncio.run(collect(...))，再 score(...)），
    避免 ragas 自带的事件循环和调用方冲突。
    """
    eval_llm = LangchainLLMWrapper(llm)
    eval_embeddings = LangchainEmbeddingsWrapper(embeddings)
    if metrics is None:
        metrics = [
            AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings),
            Faithfulness(llm=eval_llm),
            ContextRecall(llm=eval_llm),
            ContextPrecision(llm=eval_llm),
        ]

    df = pd.DataFrame(
        {
            "user_input": [s.question for s in samples],
            "retrieved_contexts": [s.contexts for s in samples],
            "response": [s.answer for s in samples],
            "reference": [s.reference for s in samples],
        }
    )
    return evaluate(
        dataset=EvaluationDataset.from_pandas(df),
        metrics=metrics,
        run_config=RunConfig(max_workers=4, timeout=900),
    )
