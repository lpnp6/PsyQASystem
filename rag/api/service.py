import rag.RAG as RAG
import configparser
import os
import asyncio
import json
import uuid
from rag.data import Document, Index
from rag.prompt import system_prompt_knowledge_extraction


config = configparser.ConfigParser()
config.read("d:/PsyQASystem/rag/api/config.ini", encoding="utf-8")

NEO4J_URI = config.get("NEO4J", "URI", fallback=None)
NEO4J_USERNAME = config.get("NEO4J", "USERNAME", fallback=None)
NEO4J_PASSWORD = config.get("NEO4J", "PASSWORD", fallback=None)
if NEO4J_URI:
    os.environ["NEO4J_URI"] = NEO4J_URI
    os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
    os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

ARK_API_KEY = config.get("ARK", "API_KEY", fallback=None)
os.environ["ARK_APIKEY"] = ARK_API_KEY

rag = RAG.RAG(
    llm_model_name="Ark", system_prompt=system_prompt_knowledge_extraction, json_format=True
)


async def build(data_path):
    documents = []
    with open(data_path, "r", encoding="utf-8") as f:
        js = json.load(f)
        documents = [
            Document(
                id="document" + str(uuid.uuid4()),
                title=document.get("Title"),
                text=document.get("Text"),
                token=document.get("Token"),
            )
            for document in js
        ]
    await rag.build_kg_batch(documents=documents,batch_size=2)


if __name__ == "__main__":
    asyncio.run(build("D:\PsyQASystem\data\\nlpxinlixue.json"))
