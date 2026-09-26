import os
import argparse
import json
import logging
from typing import Dict, List
from dotenv import load_dotenv
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Import HippoRAG components after setting environment
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hipporag_processing.log")
    ]
)

def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """Group questions by their source"""
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions

def split_text(
    text: str, 
    tokenizer: AutoTokenizer, 
    chunk_token_size: int = 256, 
    chunk_overlap_token_size: int = 32
) -> List[str]:
    """Split text into chunks based on token length with overlap"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + chunk_token_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(
            chunk_tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_token_size - chunk_overlap_token_size
    return chunks

def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    model_name: str,
    embed_model_path: str,
    llm_base_url: str,
    llm_api_key: str,
    questions: List[dict],
    sample: int
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")
    
    # Prepare output directory
    output_dir = f"./results/hipporag2/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")
    
    # Initialize tokenizer for text splitting
    try:
        tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
        logging.info(f"✅ Loaded tokenizer: {embed_model_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load tokenizer: {e}")
        return
    
    # Split text into chunks
    chunks = split_text(context, tokenizer)
    logging.info(f"✂️ Split corpus into {len(chunks)} chunks")
    
    # Format chunks as documents
    docs = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
    
    # Get questions for this corpus
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"⚠️ No questions found for corpus: {corpus_name}")
        return
    
    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]
    
    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")
    
    # Prepare queries and gold answers
    all_queries = [q["question"] for q in corpus_questions]
    gold_answers = [[q['answer']] for q in corpus_questions]
    
    # Configure HippoRAG
    config = BaseConfig(
        save_dir=os.path.join(base_dir, corpus_name),
        llm_base_url=llm_base_url,
        llm_name=model_name,
        embedding_model_name=embed_model_path.split('/')[-1],
        force_index_from_scratch=True,
        force_openie_from_scratch=True,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=5,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(docs),
        openie_mode="online"
    )
    
    # Initialize HippoRAG
    hipporag = HippoRAG(global_config=config)
    
    # Index the corpus content
    hipporag.index(docs)
    logging.info(f"✅ Indexed corpus: {corpus_name}")
    
    # Process questions
    results = []

    queries_solutions, _, _, _, _ = hipporag.rag_qa(queries=all_queries, gold_docs=None, gold_answers=gold_answers)
    solutions = [query.to_dict() for query in queries_solutions]
    
    for question in corpus_questions:
        solution = next((sol for sol in solutions if sol['question'] == question['question']), None)
        if solution:
            results.append({
                "id": question["id"],
                "question": question["question"],
                "source": corpus_name,
                "context": solution.get("docs", ""),
                "evidence": question.get("evidence", ""),
                "question_type": question.get("question_type", ""),
                "generated_answer": solution.get("answer", ""),
                "gold_answer": question.get("answer", "")
            })
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")

def main():
    # Define subset paths
    SUBSET_PATHS = {
        "medical": {
            "corpus": "./Datasets/Corpus/medical.json",
            "questions": "./Datasets/Questions/medical_questions.json"
        },
        "novel": {
            "corpus": "./Datasets/Corpus/novel.json",
            "questions": "./Datasets/Questions/novel_questions.json"
        }
    }
    
    parser = argparse.ArgumentParser(description="HippoRAG: Process Corpora and Answer Questions")
    
    # Core arguments
    parser.add_argument("--subset", required=True, choices=["medical", "novel"], 
                        help="Subset to process (medical or novel)")
    parser.add_argument("--base_dir", default="./hipporag2_workspace", 
                        help="Base working directory for HippoRAG")
    
    # Model configuration
    parser.add_argument("--model_name", default="gpt-4o-mini", 
                        help="LLM model identifier")
    parser.add_argument("--embed_model_path", default="/home/xzs/data/model/contriever", 
                        help="Path to embedding model directory")
    parser.add_argument("--sample", type=int, default=None, 
                        help="Number of questions to sample per corpus")
    
    # API configuration
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1", 
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="", 
                        help="API key for LLM service (can also use OPENAI_API_KEY environment variable)")

    args = parser.parse_args()
    
    logging.info(f"🚀 Starting HippoRAG processing for subset: {args.subset}")
    
    # Validate subset
    if args.subset not in SUBSET_PATHS:
        logging.error(f"❌ Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return
    
    # Get file paths for this subset
    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]
    
    # Handle API key security
    api_key = args.llm_api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logging.warning("⚠️ No API key provided! Requests may fail.")
    
    # Create workspace directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Load corpus data
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
        logging.info(f"📖 Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load corpus: {e}")
        return
    
    # Sample corpus data if requested
    if args.sample:
        corpus_data = corpus_data[:1]
    
    # Load question data
    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            question_data = json.load(f)
        grouped_questions = group_questions_by_source(question_data)
        logging.info(f"❓ Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load questions: {e}")
        return
    
    # Process each corpus in the subset
    for item in corpus_data:
        corpus_name = item["corpus_name"]
        context = item["context"]
        process_corpus(
            corpus_name=corpus_name,
            context=context,
            base_dir=args.base_dir,
            model_name=args.model_name,
            embed_model_path=args.embed_model_path,
            llm_base_url=args.llm_base_url,
            llm_api_key=api_key,
            questions=grouped_questions,
            sample=args.sample
        )

if __name__ == "__main__":
    main()