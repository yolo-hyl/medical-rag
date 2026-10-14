"""用 ragas 给 EvalSample 打分"""
from typing import Optional, Sequence

import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

from ..runner import EvalSample


def score(
    samples: Sequence[EvalSample],
    llm: BaseChatModel,
    embeddings: Embeddings,
    metrics: Optional[list] = None,
) -> EvaluationResult:
    """同步函数，需要在事件循环之外调用（先 asyncio.run(collect(...))，再 score(...)），
    避免 ragas 自带的事件循环和调用方冲突。

    llm / embeddings 可以用 medrag.llm.langchain_adapter 的 create_chat_model / create_embeddings 构造。
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
