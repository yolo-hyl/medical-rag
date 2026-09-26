from rouge_score import rouge_scorer
import asyncio

async def compute_rouge_score(
    answer: str,
    ground_truth: str,
    rouge_type: str = "rougeL",
    mode: str = "fmeasure"
) -> float:
    """
    Compute ROUGE score between generated answer and ground truth reference.
    
    Args:
        answer: Generated response text
        ground_truth: Reference ground truth text
        llm: Placeholder for LLM interface compatibility (not used)
        callbacks: Placeholder for callbacks (not used)
        max_retries: Placeholder for retry logic (not used)
        rouge_type: Type of ROUGE metric ('rouge1', 'rouge2', 'rougeL')
        mode: Scoring mode ('fmeasure', 'precision', 'recall')
    
    Returns:
        ROUGE score between 0.0 and 1.0
    """
    # Handle edge cases with empty texts
    if not ground_truth.strip() or not answer.strip():
        return 0.0
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    
    # Compute ROUGE score
    scores = scorer.score(ground_truth, answer)
    return getattr(scores[rouge_type], mode)