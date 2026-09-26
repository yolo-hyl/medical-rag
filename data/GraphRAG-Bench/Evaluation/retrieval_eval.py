import asyncio
import argparse
import json
import numpy as np
import os
from typing import Dict, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from .metrics import compute_context_relevance, compute_context_recall

async def evaluate_dataset(
    dataset: Dataset,
    llm: BaseLanguageModel,
    embeddings: Embeddings
) -> Dict[str, float]:
    """
    Evaluate context relevance and context recall for a dataset
    
    Args:
        dataset: Dataset containing questions, answers, contexts, and ground truths
        llm: Language model for evaluation
        embeddings: Embeddings model for semantic analysis
    
    Returns:
        Dictionary with average scores for both metrics
    """
    results = {
        "context_relevancy": [],
        "context_recall": []
    }
    
    questions = dataset["question"]
    answers = dataset["answer"]
    contexts_list = dataset["contexts"]
    ground_truths = dataset["ground_truth"]
    
    # Evaluate all samples in parallel
    tasks = []
    for i in range(len(dataset)):
        task = asyncio.create_task(
            evaluate_sample(
                question=questions[i],
                answer=answers[i],
                contexts=contexts_list[i],
                ground_truth=ground_truths[i],
                llm=llm,
                embeddings=embeddings
            )
        )
        tasks.append(task)
    
    sample_results = await asyncio.gather(*tasks)
    
    # Aggregate results
    for sample in sample_results:
        for metric, score in sample.items():
            if not np.isnan(score):  # Skip invalid scores
                results[metric].append(score)
    
    # Calculate average scores
    return {
        "context_relevancy": np.nanmean(results["context_relevancy"]),
        "context_recall": np.nanmean(results["context_recall"])
    }

async def evaluate_sample(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    llm: BaseLanguageModel,
    embeddings: Embeddings
) -> Dict[str, float]:
    """
    Evaluate context relevance and context recall for a single sample
    
    Args:
        question: User question
        answer: Generated answer
        contexts: Retrieved contexts
        ground_truth: Reference answer
        llm: Language model for evaluation
        embeddings: Embeddings model for semantic analysis
    
    Returns:
        Dictionary with scores for both metrics
    """
    # Evaluate both metrics in parallel
    relevance_task = asyncio.create_task(
        compute_context_relevance(question, contexts, llm)
    )
    
    recall_task = asyncio.create_task(
        compute_context_recall(question, contexts, ground_truth, llm)
    )
    
    # Wait for both tasks to complete
    relevance_score, recall_score = await asyncio.gather(relevance_task, recall_task)
    
    return {
        "context_relevancy": relevance_score,
        "context_recall": recall_score
    }

def parse_arguments():
    """Parse command-line arguments for evaluation configuration"""
    parser = argparse.ArgumentParser(description='RAG Evaluation Script')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to evaluation data file')
    parser.add_argument('--llm_model', type=str, required=True,
                        help='LLM model name for evaluation')
    parser.add_argument('--embedding_model', type=str, required=True,
                        help='Embedding model name')
    parser.add_argument('--base_url', type=str, default=None,
                        help='Base URL for API endpoint (optional)')
    parser.add_argument('--question_types', nargs='+', default=['type1', 'type2', 'type3', 'type4'],
                        help='List of question types to evaluate')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples per question type to evaluate')
    return parser.parse_args()

async def main():
    args = parse_arguments()
    
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Initialize models
    llm = ChatOpenAI(
        model=args.llm_model,
        base_url=args.base_url,
        api_key=api_key
    )
    
    # Initialize embeddings
    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name=args.embedding_model
    )
    embedding = LangchainEmbeddingsWrapper(embeddings=bge_embeddings)
    
    # Load evaluation data
    with open(args.data_path, 'r') as f:
        file_data = json.load(f)
    
    all_results = {}
    
    # Evaluate each question type
    for question_type in args.question_types:
        if question_type not in file_data:
            print(f"Warning: Question type '{question_type}' not found in data file")
            continue
            
        print(f"\nEvaluating question type: {question_type}")
        
        # Prepare data
        questions = [item['question'] for item in file_data[question_type][:args.num_samples]]
        ground_truths = [item['gold_answer'] for item in file_data[question_type][:args.num_samples]]
        answers = [item['generated_answer'] for item in file_data[question_type][:args.num_samples]]
        contexts = [item['context'] for item in file_data[question_type][:args.num_samples]]
        
        # Create dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)
        
        # Evaluate
        results = await evaluate_dataset(
            dataset=dataset,
            llm=llm, 
            embeddings=embedding  
        )
        
        all_results[question_type] = results
        print(f"Results for {question_type}:")
        print(f"  Context Relevance: {results['context_relevancy']:.4f}")
        print(f"  Context Recall: {results['context_recall']:.4f}")
    
    # Save final results
    print("\nFinal Evaluation Summary:")
    for q_type, metrics in all_results.items():
        print(f"\nQuestion Type: {q_type}")
        print(f"  Context Relevance: {metrics['context_relevancy']:.4f}")
        print(f"  Context Recall: {metrics['context_recall']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())