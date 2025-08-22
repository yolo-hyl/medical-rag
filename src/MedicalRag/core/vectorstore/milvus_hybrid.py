# milvus/milvus_hybrid.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient
from ...config.default_cfg import AppCfg
from .milvus_query import hybrid_search

class HybridRetriever:
    def __init__(self, client: MilvusClient, cfg: AppCfg, embedder, vectorizer):
        self.client = client
        self.cfg = cfg
        self.embedder = embedder
        self.vectorizer = vectorizer

    def _prefix(self, texts: List[str], p: Optional[str]) -> List[str]:
        return [f"{p} {t}" if p else t for t in texts]

    def prepare_query_channels(self, queries: List[str]) -> Dict[str, List[Any]]:
        out: Dict[str, List[Any]] = {}
        pre = self.cfg.embedding.dense.prefixes
        q_pref = pre.get("query") or ""
        d_pref = pre.get("document") or ""

        for ch in self.cfg.milvus.search.channels:
            if not ch.enabled:
                continue
            if ch.kind == "dense_query":
                out[ch.name] = self.embedder.embed_documents(self._prefix(queries, q_pref))
            elif ch.kind == "dense_document":
                out[ch.name] = self.embedder.embed_documents(self._prefix(queries, d_pref))
            elif ch.kind in ("sparse_query","sparse_document"):
                avgdl = 1.0
                sparse_list = []
                for q in queries:
                    toks = self.vectorizer.tokenize(q)
                    sp = self.vectorizer.build_sparse_vec_from_tokens(toks, avgdl, update_vocab=False)
                    if self.cfg.embedding.sparse_bm25.prune_empty_sparse and not sp:
                        sp = self.cfg.embedding.sparse_bm25.empty_sparse_fallback
                    sparse_list.append(sp)
                out[ch.name] = sparse_list
            else:
                raise ValueError(f"unknown channel kind: {ch.kind}")
        return out

    def search(
        self,
        queries: List[str],
        expr_vars: Optional[Dict[str, Any]] = None,
        limit_override: Optional[int] = None
    ):
        req_data = self.prepare_query_channels(queries)
        return hybrid_search(
            client=self.client,
            cfg=self.cfg,
            req_data=req_data,
            expr_vars=expr_vars,
            limit_override=limit_override
        )
