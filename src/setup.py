import os

from setuptools import setup, find_packages

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))


INSTALL_REQUIRES = [
  "langchain>=0.2.0",
  "langchain-openai>=0.1.0",
  "pymilvus>=2.4.0",
  "pydantic>=2.5",
  "fastapi>=0.110",
  "uvicorn[standard]>=0.23",
  "typer>=0.12",
  "python-dotenv>=1.0",
  "tqdm>=4.66",
  "rich>=13.7",
  "tomli; python_version<'3.11'",
]

# Installation operation
setup(
    name="MedicalRag",
    packages=["MedicalRag"],
    author="weihua li",
    url="https://github.com/yolo-hyl/medical-rag",
    version="0.1.0",
    description="Medical RAG system using LangChain + Milvus",
    keywords=["RAG", "Medical", "LangChain"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)