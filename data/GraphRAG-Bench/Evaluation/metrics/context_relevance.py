from typing import List, Callable, Awaitable
import numpy as np
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import Callbacks

CONTEXT_RELEVANCE_PROMPT = """
### Task
Evaluate the relevance of the Context for answering the Question using ONLY the information provided.
Respond ONLY with a number from 0-2. Do not explain.

### Rating Scale
0: Context has NO relevant information
1: Context has PARTIAL relevance
2: Context has RELEVANT information

### Question
{question}

### Context
{context}

### Rating:
"""

async def compute_context_relevance(
    question: str,
    contexts: List[str],
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 3
) -> float:
    """
    Evaluate the relevance of retrieved contexts for answering a question.
    Returns a score between 0.0 (irrelevant) and 1.0 (fully relevant).
    """
    # Handle edge cases
    if not question.strip() or not contexts or not any(c.strip() for c in contexts):
        return 0.0
    
    context_str = "\n".join(contexts)[:7000]  # Truncate long contexts
    
    # Check for exact matches (often indicate low relevance)
    if context_str.strip() == question.strip() or context_str.strip() in question:
        return 0.0
    
    # Get two independent ratings from LLM
    rating1 = await _get_llm_rating(question, context_str, llm, callbacks, max_retries)
    rating2 = await _get_llm_rating(question, context_str, llm, callbacks, max_retries)
    
    # Process ratings (0-2 scale) and convert to 0-1 scale
    scores = [r/2 for r in [rating1, rating2] if r is not None]
    
    # Calculate final score
    if not scores:
        return np.nan
    return sum(scores) / len(scores)  # Average of valid scores

async def _get_llm_rating(
    question: str,
    context: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int
) -> float:
    """Get a single relevance rating from LLM with retries"""
    prompt = CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context)
    
    for _ in range(max_retries):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            return _parse_rating(response.content)
        except Exception:
            continue
    return None  # Return None after max retries

def _parse_rating(text: str) -> float:
    """Extract rating from LLM response"""
    # Look for first number 0-2 in the response
    for token in text.split()[:8]:  # Check first 8 tokens
        if token.isdigit() and 0 <= int(token) <= 2:
            return float(token)
    return None  # No valid rating found