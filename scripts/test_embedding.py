import asyncio
from MedicalRag.config.default_cfg import load_cfg
from MedicalRag.core.llm.HttpClient import OllamaClient

async def test_single_embedding():
    cfg = load_cfg("config/rag.yaml")
    
    config = {
        "model_name": cfg.embedding.dense.model,
        "ollama": {
            "base_url": cfg.embedding.dense.base_url,
            "timeout": cfg.embedding.dense.timeout,
            "max_concurrent": 1,  # 先用1个并发测试
            "max_retries": 1,
        }
    }
    
    client = OllamaClient(config)
    
    try:
        print("开始测试单个嵌入...")
        result = await client.embedding("测试文本", prefix="search_document:")
        print(f"嵌入结果: {type(result)}, 长度: {len(result) if isinstance(result, list) else 'Not a list'}")
        if result:
            print(f"前5个值: {result[:5]}")
        else:
            print("嵌入结果为空！")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.aclose()

if __name__ == "__main__":
    asyncio.run(test_single_embedding())