import os
import json
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.wsa.v20250508 import wsa_client, models
from typing import List
import traceback
import re
from langchain_core.documents import Document

# 方法1: 过滤所有HTML标签（最常用）
def remove_all_html_tags(text):
    """移除所有HTML标签"""
    pattern = r'<[^>]*>'
    return re.sub(pattern, '', text)


def tencent_cloud_search(query: str, cnt: int) -> List[Document]:
    try:
        cred = credential.Credential(os.getenv("TENCENTCLOUD_SECRET_ID"), os.getenv("TENCENTCLOUD_SECRET_KEY"))
        httpProfile = HttpProfile()
        httpProfile.endpoint = "wsa.tencentcloudapi.com"
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

    except TencentCloudSDKException as err:
        print(traceback(err))
        return []
        

if __name__ == "__main__":
    docs = tencent_cloud_search(query="西瓜的药用", cnt=10)
    print(docs)