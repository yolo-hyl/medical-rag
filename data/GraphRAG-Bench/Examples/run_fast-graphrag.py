import asyncio
import os
import logging
import argparse
import json
from typing import Dict, List
from dotenv import load_dotenv
from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAILLMService, HuggingFaceEmbeddingService
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration constants
DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."
EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
]
ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]

def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """Group questions by their source"""
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions

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
    output_dir = f"./results/fast-graphrag/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")
    
    # Initialize embedding model
    try:
        embedding_tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
        embedding_model = AutoModel.from_pretrained(embed_model_path)
        logging.info(f"✅ Loaded embedding model: {embed_model_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load embedding model: {e}")
        return
    
    # Initialize GraphRAG
    grag = GraphRAG(
        working_dir=os.path.join(base_dir, corpus_name),
        domain=DOMAIN,
        example_queries="\n".join(EXAMPLE_QUERIES),
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(
            llm_service=OpenAILLMService(
                model=model_name,
                base_url=llm_base_url,
                api_key=llm_api_key,
            ),
            embedding_service=HuggingFaceEmbeddingService(
                model=embedding_model,
                tokenizer=embedding_tokenizer,
                embedding_dim=1024,
                max_token_size=8192
            ),
        ),
    )
    
    # Index the corpus content
    grag.insert(context)
    logging.info(f"✅ Indexed corpus: {corpus_name} ({len(context.split())} words)")
    
    # Get questions for this corpus
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"⚠️ No questions found for corpus: {corpus_name}")
        return
    
    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]
    
    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")
    
    # Process questions
    results = []
    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        try:
            # Execute query
            response = grag.query(q["question"])
            context_chunks = response.to_dict()['context']['chunks']
            contexts = [item[0]["content"] for item in context_chunks]
            predicted_answer = response.response

            # Collect results
            results.append({
                "id": q["id"],
                "question": q["question"],
                "source": corpus_name,
                "context": contexts,
                "evidence": q.get("evidence", ""),
                "question_type": q.get("question_type", ""),
                "generated_answer": predicted_answer,
                "gold_answer": q.get("answer", "")
            })
        except Exception as e:
            logging.error(f"❌ Error processing question {q.get('id')}: {e}")
            results.append({
                "id": q["id"],
                "error": str(e)
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
    
    parser = argparse.ArgumentParser(description="GraphRAG: Process Corpora and Answer Questions")
    
    # Core arguments
    parser.add_argument("--subset", required=True, choices=["medical", "novel"], 
                        help="Subset to process (medical or novel)")
    parser.add_argument("--base_dir", default="./Examples/graphrag_workspace", 
                        help="Base working directory for GraphRAG")
    
    # Model configuration
    parser.add_argument("--model_name", default="qwen2.5-14b-instruct", 
                        help="LLM model identifier")
    parser.add_argument("--embed_model_path", default="/home/xzs/data/model/bge-large-en-v1.5", 
                        help="Path to embedding model directory")
    parser.add_argument("--sample", type=int, default=None, 
                        help="Number of questions to sample per corpus")
    
    # API configuration
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1", 
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="", 
                        help="API key for LLM service (can also use LLM_API_KEY environment variable)")

    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"graphrag_{args.subset}.log")
        ]
    )
    
    logging.info(f"🚀 Starting GraphRAG processing for subset: {args.subset}")
    
    # Validate subset
    if args.subset not in SUBSET_PATHS:
        logging.error(f"❌ Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return
    
    # Get file paths for this subset
    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]
    
    # Handle API key security
    api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
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