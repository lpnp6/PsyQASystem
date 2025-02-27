from dataclasses import dataclass
from rag.data import Node, Edge, Index
from typing import AsyncGenerator


@dataclass
class Handler:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        pass

    async def query_done_callback(self):
        pass


@dataclass
class GraphDatabaseHandler(Handler):
    async def insert_node(self, node: Node):
        raise NotImplementedError

    async def insert_edge(self, node1: Node, node2: Node, edge: Edge):
        raise NotImplementedError


@dataclass
class VectorDatabaseHandler(Handler):
    async def insert_index(self, indices: list[Index]):
        raise NotImplementedError
    async def get_index(self, index: Index ):
        raise NotImplementedError


@dataclass
class BaseLLM:
    model_name: str
    config: dict

    async def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    async def agenerate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        raise NotImplementedError

