import json
import numpy as np
from typing import List, Dict, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import Callbacks

FACT_EXTRACTION_PROMPT = """
### Task
Extract distinct factual statements from the reference answer that could be independently verified.
Respond ONLY with a JSON object containing a "facts" list of strings.

### Example
Input:
  Question: "What causes seasons?"
  Reference: "Seasonal changes result from Earth's axial tilt. This tilt causes different hemispheres to receive varying sunlight."

Output:
{{
  "facts": [
    "Seasonal changes result from Earth's axial tilt",
    "The axial tilt causes different hemispheres to receive varying sunlight"
  ]
}}

### Actual Input
Question: "{question}"
Reference Answer: "{reference}"

### Your Response:
"""

FACT_COVERAGE_PROMPT = """
### Task
For each factual statement from the reference, determine if it's covered in the response.
Respond ONLY with a JSON object containing a "classifications" list. Each item should have:
- "statement": the exact fact from reference
- "attributed": 1 if covered, 0 if not

### Example
Response: "Seasons are caused by Earth's tilted axis"
Reference Facts: [
  "Seasonal changes result from Earth's axial tilt",
  "The axial tilt causes different hemispheres to receive varying sunlight"
]

Output:
{{
  "classifications": [
    {{"statement": "Seasonal changes result from Earth's axial tilt", "attributed": 1}},
    {{"statement": "The axial tilt causes different hemispheres to receive varying sunlight", "attributed": 0}}
  ]
}}

### Actual Input
Question: "{question}"
Response: "{response}"
Reference Facts: {facts}

### Your Response:
"""

async def compute_coverage_score(
    question: str,
    reference: str,
    response: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 2
) -> float:
    """
    Calculate coverage score (0.0-1.0) by measuring what percentage of 
    reference facts are covered in the response.
    """
    # Handle edge cases
    if not reference.strip():
        return 1.0  # Perfect coverage for empty reference
    
    # Step 1: Extract facts from reference
    facts = await _extract_facts(
        question, reference, llm, callbacks, max_retries
    )
    
    if not facts:
        return np.nan  # Failed to extract facts
    
    # Step 2: Check fact coverage in response
    coverage = await _check_fact_coverage(
        question, facts, response, llm, callbacks, max_retries
    )
    
    # Calculate coverage score
    if coverage:
        attributed = [c["attributed"] for c in coverage]
        return sum(attributed) / len(attributed)
    return np.nan

async def _extract_facts(
    question: str,
    reference: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int
) -> List[str]:
    """Extract factual statements from reference answer"""
    prompt = FACT_EXTRACTION_PROMPT.format(
        question=question,
        reference=reference[:3000]  # Truncate long references
    )
    
    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            data = json.loads(response.content)
            return _validate_facts(data.get("facts", []))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return []  # Return empty list after max retries

def _validate_facts(facts: List) -> List[str]:
    """Ensure facts are valid strings"""
    return [str(f) for f in facts if f and str(f).strip()]

async def _check_fact_coverage(
    question: str,
    facts: List[str],
    response: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int
) -> List[Dict]:
    """Check which facts are covered in the response"""
    prompt = FACT_COVERAGE_PROMPT.format(
        question=question,
        response=response[:3000],  # Truncate long responses
        facts=json.dumps(facts)
    )
    
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
            if ("statement" in item and 
                "attributed" in item and item["attributed"] in {0, 1}):
                valid.append({
                    "statement": str(item["statement"]),
                    "attributed": int(item["attributed"])
                })
        except (TypeError, ValueError):
            continue
    return valid