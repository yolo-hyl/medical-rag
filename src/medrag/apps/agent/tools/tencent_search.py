"""腾讯云联网搜索（可选依赖 tencentcloud-sdk-python，凭据从 TENCENTCLOUD_SECRET_ID / KEY 读取）"""
import json
import logging
import os
import re
from typing import List

from langchain_core.documents import Document
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.wsa.v20250508 import models, wsa_client

logger = logging.getLogger(__name__)


def remove_all_html_tags(text):
    return re.sub(r'<[^>]*>', '', text)


def tencent_cloud_search(query: str, cnt: int) -> List[Document]:
    try:
        cred = credential.Credential(os.getenv("TENCENTCLOUD_SECRET_ID"), os.getenv("TENCENTCLOUD_SECRET_KEY"))
        client = wsa_client.WsaClient(cred, "")

        # 实例化一个请求对象,每个接口都会对应一个request对象
        req = models.SearchProRequest()
        params = {
            "Query": query,
            "Cnt": cnt if cnt > 10 else 10
        }
        req.from_json_string(json.dumps(params))
        
        # 返回的resp是一个SearchProResponse的实例，与请求对象对应
        resp = client.SearchPro(req).Pages
        # 输出json格式的字符串回包
        docs = []
        for i in resp:
            obj = json.loads(i)
            doc = Document(page_content=remove_all_html_tags(obj["passage"]), metadata=obj)
            if len(docs) >= cnt:
                break
            docs.append(doc)
        return docs

    except TencentCloudSDKException:
        logger.exception("腾讯云联网搜索失败")
        return []
        

