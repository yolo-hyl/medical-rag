import os

import uvicorn

if __name__ == "__main__":
    # API keys must be set in the environment before starting.
    # Example: export DASHSCOPE_API_KEY=sk-...
    uvicorn.run(
        "MedicalRag.api.app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        workers=1,       # MUST be 1: session state lives in process memory
        log_level="info",
        reload=False,
    )
