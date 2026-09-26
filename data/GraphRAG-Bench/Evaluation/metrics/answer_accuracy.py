import asyncio
import json
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.callbacks import Callbacks

# Define necessary Pydantic models
class StatementsWithReason(BaseModel):
    statement: str
    reason: str

class ClassificationWithReason(BaseModel):
    TP: List[StatementsWithReason] = []
    FP: List[StatementsWithReason] = []
    FN: List[StatementsWithReason] = []

class QuestionAnswerGroundTruth(BaseModel):
    question: str
    answer: List[str]
    ground_truth: List[str]

# F-beta score calculation
def fbeta_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)

# Statement generation prompt template
STATEMENT_GENERATOR_PROMPT = """
Generate concise independent statements from the given text that represent factual claims.
Respond ONLY with a JSON array of strings. Do not include any other text.

Example Input: 
"The sun is powered by nuclear fusion. This process creates light and heat."

Example Output:
["The sun is powered by nuclear fusion", "Nuclear fusion creates light and heat"]

Input Text:
{text}

Generated Statements:
"""

# Correctness classification prompt template
CORRECTNESS_PROMPT_TEMPLATE = """
Analyze statements from an answer compared to ground truth. Classify each as:
- TP (True Positive): Present in answer and supported by ground truth
- FP (False Positive): Present in answer but unsupported
- FN (False Negative): Missing from answer but present in ground truth

Provide JSON output with lists of TP, FP, FN objects containing 'statement' and 'reason'.

Examples:
{examples}

Current Analysis:
Question: "{question}"
Answer Statements: {answer}
Ground Truth Statements: {ground_truth}
"""

# Pre-defined examples for correctness classification
CORRECTNESS_EXAMPLES = [
    {
        "input": {
            "question": "What powers the sun and its primary function?",
            "answer": [
                "The sun is powered by nuclear fission",
                "Its primary function is providing light"
            ],
            "ground_truth": [
                "The sun is powered by nuclear fusion",
                "Fusion creates energy for heat and light",
                "Sunlight is essential for Earth's climate"
            ]
        },
        "output": {
            "TP": [{"statement": "Its primary function is providing light", "reason": "Matches ground truth about light"}],
            "FP": [{"statement": "The sun is powered by nuclear fission", "reason": "Contradicts fusion fact"}],
            "FN": [
                {"statement": "The sun is powered by nuclear fusion", "reason": "Missing correct power source"},
                {"statement": "Fusion creates energy for heat and light", "reason": "Missing energy creation detail"}
            ]
        }
    }
]

async def compute_answer_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    weights: List[float] = [0.75, 0.25],
    beta: float = 1.0,
    callbacks: Callbacks = None
) -> float:
    """Compute answer correctness score combining factuality and semantic similarity"""
    # Generate statements from answer and ground truth
    answer_statements = await generate_statements(llm, answer, callbacks)
    gt_statements = await generate_statements(llm, ground_truth, callbacks)

    # Calculate factuality score using statement classification
    factuality_score = await calculate_factuality(
        llm, question, answer_statements, gt_statements, callbacks, beta
    ) if weights[0] != 0 else 0.0

    # Calculate semantic similarity
    similarity_score = await calculate_semantic_similarity(
        embeddings, answer, ground_truth
    ) if weights[1] != 0 else 0.0

    # Combine scores using weighted average
    return float(np.average([factuality_score, similarity_score], weights=weights))

async def generate_statements(
    llm: BaseLanguageModel, text: str, callbacks: Callbacks
) -> List[str]:
    """Generate concise factual statements from text"""
    prompt = STATEMENT_GENERATOR_PROMPT.format(text=text)
    response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return []

async def calculate_factuality(
    llm: BaseLanguageModel,
    question: str,
    answer_stmts: List[str],
    gt_stmts: List[str],
    callbacks: Callbacks,
    beta: float
) -> float:
    """Classify statements and calculate factuality F-beta score"""
    if not answer_stmts and not gt_stmts:
        return 1.0  # Perfect score if both empty

    # Prepare examples for prompt
    examples = "\n".join(
        f"Input: {json.dumps(ex['input'])}\nOutput: {json.dumps(ex['output'])}"
        for ex in CORRECTNESS_EXAMPLES
    )

    # Generate classification
    prompt = CORRECTNESS_PROMPT_TEMPLATE.format(
        examples=examples,
        question=question,
        answer=json.dumps(answer_stmts),
        ground_truth=json.dumps(gt_stmts)
    )
    response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
    
    try:
        classification = ClassificationWithReason(**json.loads(response.content))
        tp = len(classification.TP)
        fp = len(classification.FP)
        fn = len(classification.FN)
        return fbeta_score(tp, fp, fn, beta)
    except (json.JSONDecodeError, TypeError):
        return 0.0  # Return minimum score on failure

async def calculate_semantic_similarity(
    embeddings: Embeddings, answer: str, ground_truth: str
) -> float:
    """Compute cosine similarity between answer and ground truth embeddings"""
    a_embed, gt_embed = await asyncio.gather(
        embeddings.aembed_query(answer),
        embeddings.aembed_query(ground_truth)
    )
    cosine_sim = np.dot(a_embed, gt_embed) / (
        np.linalg.norm(a_embed) * np.linalg.norm(gt_embed))
    return (cosine_sim + 1) / 2  # Scale to [0, 1]