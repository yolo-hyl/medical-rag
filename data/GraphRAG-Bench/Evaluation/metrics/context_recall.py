import json
import numpy as np
from typing import List, Dict, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import Callbacks

CONTEXT_RECALL_PROMPT = """
### Task
Analyze each sentence in the Answer and determine if it can be attributed to the Context.
Respond ONLY with a JSON object containing a "classifications" list. Each item should have:
- "statement": the exact sentence from Answer
- "reason": brief explanation (1 sentence)
- "attributed": 1 for yes, 0 for no

### Example
Input:
Context: "Einstein won the Nobel Prize in 1921 for physics."
Answer: "Einstein received the Nobel Prize. He was born in Germany."

Output:
{{
  "classifications": [
    {{
      "statement": "Einstein received the Nobel Prize",
      "reason": "Matches context about Nobel Prize",
      "attributed": 1
    }},
    {{
      "statement": "He was born in Germany",
      "reason": "Birth information not in context",
      "attributed": 0
    }}
  ]
}}

### Actual Input
Context: "{context}"

Answer: "{answer}"

Question: "{question}" (for reference only)

### Your Response:
"""

async def compute_context_recall(
    question: str,
    contexts: List[str],
    reference_answer: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 2
) -> float:
    """
    Calculate context recall score (0.0-1.0) by measuring what percentage of 
    reference answer statements are supported by the context.
    """
    # Handle edge cases
    if not reference_answer.strip():
        return 1.0  # Perfect recall for empty reference
    
    context_str = "\n".join(contexts)
    if not context_str.strip():
        return 0.0  # No context means no attribution
    
    # Format prompt with actual data
    prompt = CONTEXT_RECALL_PROMPT.format(
        question=question,
        context=context_str[:10000],  # Truncate long contexts
        answer=reference_answer[:2000]  # Truncate long answers
    )
    
    # Get LLM classification with retries
    classifications = await _get_classifications(
        prompt, llm, callbacks, max_retries
    )
    
    # Calculate recall score
    if classifications:
        attributed = [c["attributed"] for c in classifications]
        return sum(attributed) / len(attributed)
    return np.nan  # Return NaN if no valid classifications

async def _get_classifications(
    prompt: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int
) -> List[Dict]:
    """Get valid classifications from LLM with retries"""
    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            data = json.loads(response.content)
            return _validate_classifications(data.get("classifications", []))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return []  # Return empty list after max retries

def _validate_classifications(classifications: List) -> List[Dict]:
    """Ensure classifications have required fields and proper types"""
    valid = []
    for item in classifications:
        try:
            # Validate required fields and types
            if ("statement" in item and "reason" in item and 
                "attributed" in item and item["attributed"] in {0, 1}):
                valid.append({
                    "statement": str(item["statement"]),
                    "reason": str(item["reason"]),
                    "attributed": int(item["attributed"])
                })
        except (TypeError, ValueError):
            continue
    return valid