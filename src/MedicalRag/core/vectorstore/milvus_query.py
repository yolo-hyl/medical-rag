# milvus/milvus_query.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker
from ...config.milvus_cfg import AppCfg, ChannelCfg
import logging

def _render_expr(channel: ChannelCfg, sch, expr_vars: Optional[Dict[str, Any]]) -> str:
    tpl = (channel.expr_template or sch.expr_template or "").strip()
    if not tpl:
        return ""
    return tpl.format(**(expr_vars or {}))

def _make_req(data_list: List[Any], channel: ChannelCfg, expr: str) -> AnnSearchRequest:
    logging.info(
        f"\n创建 AnnSearchRequest Params: \n " + 
        f"anns_field: {channel.field} \n "+
        f"param.metric_type: {channel.metric_type} \n " +
        f"param.params: {channel.params or {}} \n "+
        f"limit: {channel.limit} \n "+
        f"expr: {channel.field} \n " + 
        f"weight: {channel.weight}"
    )
    return AnnSearchRequest(
        data=data_list,
        anns_field=channel.field,
        param={"metric_type": channel.metric_type, "params": channel.params or {}},
        limit=channel.limit,
        expr=expr  # ← 关键：表达式放到每个 AnnSearchRequest
    )

def _page_to_offset(page: int, page_size: int) -> int:
    page = max(1, page)
    return (page - 1) * page_size

def hybrid_search(
    client: MilvusClient,
    cfg: AppCfg,
    req_data: Dict[str, List[Any]],          # key: channel.name -> N 条数据
    expr_vars: Optional[Dict[str, Any]] = None,
    page: int = 1,
    page_size: Optional[int] = None,
    limit_override: Optional[int] = None,
    output_fields: Optional[List[str]] = None
):
    sch = cfg.milvus.search
    page_size = page_size or sch.pagination.page_size
    offset = _page_to_offset(page, page_size)
    output_fields = output_fields or sch.output_fields

    reqs = []
    for ch in sch.channels:
        if not ch.enabled:
            continue
        dl = req_data.get(ch.name)
        if not dl:
            continue
        expr = _render_expr(ch, sch, expr_vars)
        reqs.append(_make_req(dl, ch, expr))
    
    if sch.rrf.enabled:
        ranker = RRFRanker(k=sch.rrf.k)
    else:
        weights = []
        for ch in sch.channels:
            if ch.enabled:
                if ch.weight != -1 and ch.weight >= 0 and ch.weight <= 1:
                    weights.append(ch.weight)
                else:
                    raise f"使用WeightedRanker时，开启的channel配置的权重非法: {ch.name} weight: {ch.weight}"
        ranker = WeightedRanker(*weights)

    res = client.hybrid_search(
        collection_name=cfg.milvus.collection.name,
        reqs=reqs,
        ranker=ranker,
        limit=limit_override or sch.default_limit,
        offset=offset,
        output_fields=output_fields
    )
    return res
