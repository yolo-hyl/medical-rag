from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.metrics import AnswerRelevancy, Faithfulness, ContextRecall, ContextPrecision
from abc import ABC, abstractmethod
from ragas.llms import LangchainLLMWrapper
from langchain_community.chat_models import ChatOpenAI
from .RagBase import BasicRAG
from datasets import Dataset
from langchain_core.language_models import BaseChatModel
from ragas.llms import LangchainLLMWrapper
from tqdm import tqdm
import pandas as pd
from ragas import EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig


class RagEvaluateBase(ABC):
    def __init__(self, rag_components: BasicRAG, eval_datasets: Dataset, eval_llm: BaseChatModel, embedding=None) -> None:
        self.rag = rag_components
        self.eval_datasets = eval_datasets
        self.eval_llm = LangchainLLMWrapper(eval_llm)
        self.embedding = LangchainEmbeddingsWrapper(embedding)
    
    def do_sample(self, sample_data: int):
        """ 采样评测数据集 """
        self.eval_datasets = self.eval_datasets.shuffle().select(range(sample_data))
        
    @abstractmethod
    def do_evaluate(self, datasets_query_field_name: str, datasets_reference_field_name: str):
        pass
        
            
class RagasRagEvaluate(RagEvaluateBase):
    def __init__(self, rag_components: BasicRAG, eval_datasets: Dataset, eval_llm: BaseChatModel, embedding=None) -> None:
        super().__init__(rag_components, eval_datasets, eval_llm, embedding)
        
    def do_evaluate(self, datasets_query_field_name: str, datasets_reference_field_name: str) -> EvaluationResult:
        response_list = []
        retrieved_contexts_list = []
        queries_list = []
        references_list = []
        for data_item in tqdm(self.eval_datasets, desc="Answering questions"):
            rag_response = self.rag.answer(query=data_item[datasets_query_field_name], return_document=True)
            
            retrieved_contexts_list.append([document.metadata['document'] for document in rag_response["documents"]])
            response_list.append(rag_response["answer"])
            queries_list.append(str(data_item[datasets_query_field_name]))
            references_list.append(str(data_item[datasets_reference_field_name]))
        df = pd.DataFrame(
            {
                "user_input": queries_list,
                "retrieved_contexts": retrieved_contexts_list,
                "response": response_list,
                "reference": references_list,
            }
        )
        rag_results = EvaluationDataset.from_pandas(df)
        results = evaluate(
            dataset=rag_results,
            metrics=[
                AnswerRelevancy(llm=self.eval_llm, embeddings=self.embedding),
                Faithfulness(llm=self.eval_llm),
                ContextRecall(llm=self.eval_llm),
                ContextPrecision(llm=self.eval_llm),
            ],
            run_config=RunConfig(
                max_workers=4,
                timeout=900
            )
        )
        return results