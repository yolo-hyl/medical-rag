"""
把具体实现包装成统一的答题函数（AnswerFn）

其他库只要提供一个 `question -> (answer, contexts)` 的函数，就能接入同一套评测流程。
"""
from __future__ import annotations

from typing import List, Tuple

from .runner import AnswerFn


def from_medical_rag(rag, context_field: str = "document") -> AnswerFn:
    """包装 MedicalRag 的 SimpleRAG / MultiDialogueRag"""
    async def answer_fn(question: str) -> Tuple[str, List[str]]:
        result = await rag.answer(question, return_document=True)
        contexts = [
            d.metadata.get(context_field) or d.page_content
            for d in result["documents"]
        ]
        return result["answer"], contexts

    return answer_fn


def from_multi_dialogue_rag(rag, context_field: str = "document") -> AnswerFn:
    """包装多轮 RAG：每个问题用独立的 session_id，避免题与题之间共享历史"""
    counter = {"n": 0}

    async def answer_fn(question: str) -> Tuple[str, List[str]]:
        counter["n"] += 1
        result = await rag.answer(
            question, return_document=True, session_id=f"eval-{counter['n']}"
        )
        contexts = [
            d.metadata.get(context_field) or d.page_content
            for d in result["documents"]
        ]
        return result["answer"], contexts

    return answer_fn
