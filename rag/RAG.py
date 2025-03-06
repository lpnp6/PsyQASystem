from dataclasses import dataclass, field, asdict
from collections import deque
from typing import Type
from rag.data import Document, Node
from rag.base import GraphDatabaseHandler, BaseLLM
from rag.embedding import Embedding_model, Spacy
from rag.document_layer import chunk_by_sematic, insert_chunk
from rag.utils import logger, set_logger
from rag.prompt import system_prompt_knowledge_extraction
from rag.entity_relation_layer import build_llm_based
import os
import logging
import asyncio

db_handler_paths = {
    "Neo4jGraphDatabaseHandler": ".db.neo4j_db_handler",
}

llm_model_paths = {
    "DeepSeekR1": ".llm.openai.llms",
    "Spark": ".llm.spark.spark",
    "DeepSeekChat": ".llm.openai.llms",
    "SiliconFlowDeepSeekChat": ".llm.openai.llms",
    "SiliconFlowDeepSeekR1": ".llm.openai.llms",
    "Ark": ".llm.openai.llms",
    "OllamaDeepSeekR1": ".llm.openai.llms",
    "Openai4oMini": ".llm.openai.llms",
}


def lazy_external_import(module_name, class_name):
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


@dataclass
class RAG:
    log_level: int = field(default=logging.INFO)
    log_dir: str = field(default=os.getcwd())

    embedding_model_cls: Embedding_model = field(default=Spacy)
    embedding_dim: int = field(
        default=768
    )  # embedding_dim should be consistent with embedding_model

    namespace_prefix: str = field(default="")
    graph_db_handler_name: str = field(default="Neo4jGraphDatabaseHandler")

    llm_model_name: str = field(default="DeepSeekR1")
    system_prompt: str = field(default=system_prompt_knowledge_extraction)
    response_prefix: str = field(default=None)
    json_format: bool = field(default=True)

    chunking_func: callable = field(default=chunk_by_sematic)
    insert_chunk: callable = field(default=insert_chunk)

    build: callable = field(default=build_llm_based)

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        absolute_path = os.path.join(self.log_dir, "RAG.log")
        set_logger(absolute_path)
        logger.setLevel(self.log_level)
        logger.info(f"Initialize logger.")

        global_config = asdict(self)
        _sensitive_keys = {}
        _global_config = ",\n".join(
            [f"{k} = {v}" for k, v in global_config.items() if k not in _sensitive_keys]
        )
        logger.info("Initialize parameters" + _global_config)

        self.embedding_model = self.embedding_model_cls(
            embedding_dim=self.embedding_dim
        )

        self.graph_db_handler_cls: Type[GraphDatabaseHandler] = self._get_cls(
            self.graph_db_handler_name, db_handler_paths
        )
        self.graph_db_handler: GraphDatabaseHandler = self.graph_db_handler_cls(
            namespace=self.namespace_prefix + "graph_db_handler",
            global_config=global_config,
        )

        self.llm_model_cls: Type[BaseLLM] = self._get_cls(
            self.llm_model_name, llm_model_paths
        )
        self.llm_model: BaseLLM = self.llm_model_cls(
            model_name=self.llm_model_name, config=global_config
        )
        logger.info(f"LLM model {self.llm_model.model_name} api is available.")

    def _get_cls(self, name: str, paths: dict):
        path = paths[name]
        return lazy_external_import(path, name)

    async def insert_document(self, document: Document):
        try:
            chunks = await self.chunking_func(document, self.embedding_model)
            tasks = [
                asyncio.create_task(self.insert_chunk(chunk, self.graph_db_handler))
                for chunk in chunks
            ]
            await asyncio.gather(*tasks)
            logger.info(
                f"Successfully split document {document.title} into chunks and insert them into graph database."
            )
            return chunks
        except Exception as e:
            logger.error(f"{e}")
            raise e

    async def build_kg(self, chunk_nodes: list[Node]):
        try:
            document_title = chunk_nodes[0].properties["document_title"]
            results = await self.build(
                chunk_nodes, self.graph_db_handler, self.llm_model, self.embedding_model
            )
            logger.info(
                f"Successfully build knowledge graph from document {document_title}"
            )
            nodes = results["entities"]
            return nodes
        except Exception as e:
            logger.error(
                f"""Failed to extract knowledge from document {document_title} and insert into knowledge graph.\n{e}"""
            )
            raise e

    async def build_kg_batch(
        self, documents: list[Document], batch_size=256, max_attempt=3
    ):
        queue = deque((document, 0) for document in documents)
        while queue:
            documents_batch = [
                queue.popleft() for _ in range(min(len(queue), batch_size))
            ]
            tasks = []
            for document, _ in documents_batch:
                tasks.append(asyncio.create_task(self.insert_document(document)))
            chunks_batch = await asyncio.gather(*tasks, return_exceptions=True)
            kg_tasks = []
            for (document, attempt), chunks in zip(documents_batch, chunks_batch):
                if isinstance(chunks, Exception):
                    if attempt < max_attempt:
                        logger.error(
                            f"Failed to insert document {document.title} (attempt {attempt + 1}/{max_attempt}): {chunks}"
                        )
                        queue.append((document, attempt + 1))
                    else:
                        logger.info(
                            f"Document {document.title} failed after {max_attempt} attempts. Skipping."
                        )
                else:
                    kg_tasks.append(asyncio.create_task(self.build_kg(chunks)))
            if kg_tasks:
                kg_results = await asyncio.gather(*kg_tasks, return_exceptions=True)
                for (document, attempt), kg_result in zip(documents_batch, kg_results):
                    if isinstance(kg_result, Exception):
                        if attempt < max_attempt:
                            logger.error(
                                f"Failed to insert document {document.title} (attempt {attempt + 1}/{max_attempt}): {chunks}"
                            )
                            queue.append((document, attempt + 1))
                        else:
                            logger.info(
                                f"Document {document.title} failed after {max_attempt} attempts. Skipping."
                            )
