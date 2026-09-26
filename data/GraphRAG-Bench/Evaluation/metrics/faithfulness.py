import json
import numpy as np
from typing import List, Dict, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import Callbacks

STATEMENT_GENERATION_PROMPT = """
### Task
Break down the answer into atomic statements that are fully understandable without pronouns.
Respond ONLY with a JSON array of strings.

### Example
Question: "Who was Albert Einstein?"
Answer: "He was a German physicist known for relativity."
Output: ["Albert Einstein was a German physicist", "Albert Einstein is known for relativity"]

### Actual Input
Question: "{question}"
Answer: "{answer}"

### Generated Statements:
"""

FAITHFULNESS_EVALUATION_PROMPT = """
### Task
Judge if each statement can be directly inferred from the context. 
Respond ONLY with a JSON array of objects, each containing:
- "statement": the exact statement
- "verdict": 1 (supported) or 0 (not supported)
- "reason": brief explanation (1 sentence)

### Context
{context}

### Statements to Evaluate
{statements}

### Example Response
[
  {{"statement": "John is a computer science major", "verdict": 1, "reason": "Context says John studies Computer Science"}},
  {{"statement": "John works part-time", "verdict": 0, "reason": "No mention of employment in context"}}
]

### Your Response:
"""

async def compute_faithfulness_score(
    question: str,
    answer: str,
    contexts: List[str],
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 2
) -> float:
    """
    Calculate faithfulness score (0.0-1.0) by measuring what percentage of 
    answer statements are supported by the context.
    """
    # Step 1: Generate atomic statements from answer
    statements = await _generate_statements(
        question, answer, llm, callbacks, max_retries
    )
    
    # Handle edge cases
    if not statements:
        return 1.0 if not answer.strip() else np.nan
    
    context_str = "\n".join(contexts)
    if not context_str.strip():
        return 0.0  # No context means no support
    
    # Step 2: Evaluate statement faithfulness
    verdicts = await _evaluate_statements(
        statements, context_str, llm, callbacks, max_retries
    )
    
    # Calculate faithfulness score
    if verdicts:
        supported = [v["verdict"] for v in verdicts]
        return sum(supported) / len(supported)
    return np.nan

async def _generate_statements(
    question: str,
    answer: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int
) -> List[str]:
    """Break down answer into atomic statements"""
    prompt = STATEMENT_GENERATION_PROMPT.format(
        question=question[:500],  # Truncate long questions
        answer=answer[:3000]      # Truncate long answers
    )
    
    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            return json.loads(response.content)
        except json.JSONDecodeError:
            continue
    return []  # Return empty list after max retries

async def _evaluate_statements(
    statements: List[str],
    context: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int
) -> List[Dict]:
    """Evaluate which statements are supported by context"""
    prompt = FAITHFULNESS_EVALUATION_PROMPT.format(
        context=context[:10000],  # Truncate long contexts
        statements=json.dumps(statements)[:5000]  # Truncate statement list
    )
    
    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            return _validate_verdicts(json.loads(response.content))
        except (json.JSONDecodeError, TypeError):
            continue
    return []  # Return empty list after max retries

def _validate_verdicts(verdicts: List) -> List[Dict]:
    """Ensure verdicts have required fields and proper types"""
    valid = []
    for item in verdicts:
        try:
            # Validate required fields and types
            if ("statement" in item and 
                "verdict" in item and item["verdict"] in {0, 1} and
                "reason" in item):
                valid.append({
                    "statement": str(item["statement"]),
                    "verdict": int(item["verdict"]),
                    "reason": str(item["reason"])
                })
        except (TypeError, ValueError):
            continue
    return valid