import math
from typing import List, Dict, Iterable, Iterator
import pkuseg
from multiprocessing import Pool, cpu_count
import os, gzip, pickle, math
from pathlib import Path
from stopwords import stopwords, filter_stopwords

current_dir = Path(__file__).resolve().parent
default_vocab_dir = str(current_dir) + "/vocab/"

# ====== worker 全局 ======
_SEG = None  # 每个子进程里各自持有一个分词器

def _init_seg_worker(domain_model: str):
    """
    每个子进程启动时运行：加载各自的 pkuseg 实例
    """
    global _SEG
    import pkuseg as _pk  # 避免主进程/子进程导入冲突
    _SEG = _pk.pkuseg(model_name=domain_model)

def _cut_worker(text: str) -> List[str]:
    """
    子进程真正执行的分词函数
    """
    toks = filter_stopwords(_SEG.cut(text))
    return [t.strip() for t in toks if t.strip()]


class Vocabulary:
    """维护 token->id 与 id->df，用于稀疏向量化"""
    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.df: Dict[int, int] = {}
        self.N: int = 0            # 文档总数
        self.sum_dl: int = 0       # 所有文档长度之和（可选，用于 avgdl）
        # 可选：冻结后缓存
        self.idf_arr: List[float] | None = None

    def add_document(self, tokens: List[str]):
        self.N += 1
        self.sum_dl += len(tokens)
        seen = set()
        for t in tokens:
            if t not in self.token2id:
                self.token2id[t] = len(self.token2id)
            tid = self.token2id[t]
            if tid not in seen:
                self.df[tid] = self.df.get(tid, 0) + 1
                seen.add(tid)
        # 词表改变后，旧的 idf 缓存作废
        self.idf_arr = None

    def idf(self, tid: int) -> float:
        if self.idf_arr is not None:
            return self.idf_arr[tid]
        df = self.df.get(tid, 0)
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def freeze(self):
        """建库结束后一次性缓存 idf，加速查询"""
        V = len(self.token2id)
        df_arr = [0]*V
        for tid, c in self.df.items():
            df_arr[tid] = c
        self.idf_arr = [math.log(1.0 + (self.N - d + 0.5)/(d + 0.5)) for d in df_arr]

    # ---------- 保存 / 加载 ----------
    def save(self, path: str, compress: bool = True):
        state = {
            "version": 1,
            "token2id": self.token2id,
            "df": self.df,
            "N": self.N,
            "sum_dl": self.sum_dl,
            # 可选缓存，存在就一起存
            "idf_arr": self.idf_arr,
        }
        data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        
        if '/' not in path:  # 没有自定义绝对路径 , 自动保存在当前目录下的vocab文件夹
            tmp = str(default_vocab_dir) + path + ".tmp"
            path = str(default_vocab_dir) + path
        else:
            tmp = path + ".tmp"
            
        with (gzip.open(tmp, "wb") if compress else open(tmp, "wb")) as f:
            f.write(data)
        os.replace(tmp, path)  # 原子替换，防止中途写坏

    @classmethod
    def load(cls, path_or_name: str):
        
        if '/' not in path_or_name:  # 直接传入的名字
            path = str(default_vocab_dir) + "/" + path_or_name
            
        with (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")) as f:
            state = pickle.load(f)
        v = cls()
        v.token2id = state["token2id"]
        v.df = state["df"]
        v.N = state["N"]
        v.sum_dl = state.get("sum_dl", 0)
        v.idf_arr = state.get("idf_arr", None)
        return v

# ====== 并行分词 + BM25 ======
class BM25Vectorizer:
    def __init__(self, vocab: Vocabulary, domain_model: str = "medicine"):
        # 单进程下仍可直接用
        self.seg = pkuseg.pkuseg(model_name=domain_model)  
        self.domain_model = domain_model
        self.vocab = vocab
        # BM25 参数
        self.k1 = 1.5
        self.b = 0.75

    # --- 单进程分词 ---
    def tokenize(self, text: str) -> List[str]:
        return [t.strip() for t in filter_stopwords(self.seg.cut(text)) if t.strip()]

    # --- 多进程分词（批处理/流式产出）---
    def tokenize_parallel(
        self,
        texts: Iterable[str],
        workers: int = None,
        chunksize: int = 64
    ) -> Iterator[List[str]]:
        """
        并行分词：按 chunksize 批量发给子进程，流式返回 tokens 列表。
        """
        if workers is None:
            workers = max(1, cpu_count() - 1)
        with Pool(
            processes=workers,
            initializer=_init_seg_worker,
            initargs=(self.domain_model,)
        ) as pool:
            # imap 是流式的，内存占用更稳
            for tokens in pool.imap(_cut_worker, texts, chunksize=chunksize):
                yield tokens

    # --- 从 tokens 构建稀疏向量（避免重复分词） ---
    def build_sparse_vec_from_tokens(
        self,
        tokens: List[str],
        avgdl: float,
        update_vocab: bool = False
    ) -> Dict[int, float]:
        """
        允许传入已分好的 tokens（建议并行切好后再喂这里）
        """
        if update_vocab:
            self.vocab.add_document(tokens)

        # 查询阶段要容忍 OOV
        tf: Dict[int, int] = {}
        for t in tokens:
            tid = self.vocab.token2id.get(t)
            if tid is None:
                continue
            tf[tid] = tf.get(tid, 0) + 1

        if not tf:
            return {}

        dl = sum(tf.values())
        K = self.k1 * (1 - self.b + self.b * dl / max(avgdl, 1.0))

        vec: Dict[int, float] = {}
        for tid, f in tf.items():
            idf = self.vocab.idf(tid)
            score = idf * (f * (self.k1 + 1.0)) / (f + K)
            if score > 0:
                vec[tid] = float(score)
        return vec

    # --- 兼容原 API：直接给文本 ---
    def build_sparse_vec(self, text: str, avgdl: float, update_vocab: bool = False):
        tokens = self.tokenize(text)
        return self.build_sparse_vec_from_tokens(tokens, avgdl, update_vocab)
