import tiktoken

ESTIMATE_FUNCTION_REGISTRY = {}

def register_estimate_function(name):
    """装饰器：注册函数到字典"""
    def decorator(func):
        ESTIMATE_FUNCTION_REGISTRY[name] = func
        return func
    return decorator
    
@register_estimate_function("tiktoken")
def estimate_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = len(encoding.encode(text))
    return tokens
