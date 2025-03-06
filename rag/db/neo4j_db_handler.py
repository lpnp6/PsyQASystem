from neo4j import AsyncGraphDatabase
from dataclasses import dataclass
from rag.data import Node, Edge
from rag.utils import logger
from rag.base import GraphDatabaseHandler
import numpy as np
import asyncio
import uuid
import time
import os
import re


@dataclass
class Neo4jGraphDatabaseHandler(GraphDatabaseHandler):
    def __init__(self, namespace, global_config):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
        )
        URI = os.environ.get("NEO4J_URI")
        USERNAME = os.environ.get("NEO4J_USERNAME")
        PASSWORD = os.environ.get("NEO4J_PASSWORD")
        MAX_CONNECTION_POOL_SIZE = os.environ.get("NEO4J_MAX_CONNECTION_POOL_SIZE", 800)

        self._driver = AsyncGraphDatabase.driver(
            URI,
            auth=(USERNAME, PASSWORD),
            max_connection_lifetime=30 * 60,  # 30分钟重新建立连接
            connection_timeout=60,  # 60秒连接超时
            keep_alive=True,  # 启用TCP keep-alive
            connection_acquisition_timeout=120,
            max_transaction_retry_time=30,
            max_connection_pool_size=MAX_CONNECTION_POOL_SIZE,
        )

    def close(self):
        self.driver.close()

    async def insert_node(self, n: Node, max_retries=3):
        label = n.label
        query = f"""
        MERGE (n:{label} {{node_id: $node_id}})
        on CREATE SET n = $properties
        on MATCH SET n += $properties
        return n
        """
        for attempt in range(max_retries):
            try:
                async with self._driver.session() as session:
                    record = await session.run(
                        query=query,
                        node_id=n.properties["node_id"],
                        properties=n.properties,
                    )
                    node = await record.single()
                    if node is not None:
                        return n
                    else:
                        return None
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                else:
                    return e

    async def insert_edge(self, node1, node2, edge, max_retries=3):
        label = edge.label
        query = f"""MERGE (a:Entity{{node_id: $node_id1}})
                    MERGE (b:Entity{{node_id: $node_id2}})
                    MERGE (a)-[r :{label}]->(b)
                    ON CREATE SET  r = $properties
                    ON MATCH SET r += $properties
                    return r"""
        node_id1 = node1.properties["node_id"]
        node_id2 = node2.properties["node_id"]
        properties = edge.properties
        for attempt in range(max_retries):
            try:
                async with self._driver.session() as session:
                    result = await session.run(
                        query=query,
                        node_id1=node_id1,
                        node_id2=node_id2,
                        properties=properties,
                    )
                    return await result.single()
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                else:
                    return e

    async def merge(self, node: Node, max_retries=3):
        node_id = node.properties.get("node_id")
        query = f"""MATCH (new_node:Entity {{node_id: $node_id}}), (existing_node:Entity)
                    WITH new_node, existing_node,
                    gds.similarity.cosine(new_node.name_embedding, existing_node.name_embedding) AS sim_name,
                    gds.similarity.cosine(new_node.type_embedding, existing_node.type_embedding) AS sim_type,
                    gds.similarity.cosine(new_node.description_embedding, existing_node.description_embedding) AS sim_description
                    WITH new_node, existing_node, sim_name, sim_type, sim_description,
                    (sim_name * 0.6 + sim_type * 0.3 + sim_description * 0.1) AS composite_sim
                    WHERE composite_sim > 0.95 AND new_node.node_id <> existing_node.node_id AND new_node.document_id <> existing_node.document_id
                    CREATE (new_node)-[:SIMILAR]->(existing_node)
                    CREATE (existing_node)-[:SIMILAR]->(new_node)
                    RETURN new_node, existing_node"""
        for attempt in range(max_retries):
            try:
                async with self._driver.session() as session:
                    result = await session.run(query=query, node_id=node_id)
                    return await result.single()
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                else:
                    return e

    async def semantic_search(
        self, node: Node, top_k: int, max_retries=3
    ) -> tuple[Node, Edge, Node]:
        name_embedding = node.properties.get("name_embedding")
        query = f"""
        MATCH (n:Entiy)
        WITH n,
        apoc.algo.cosineSimilarity(n.name_embedding, $name_embedding) AS similarity
        ORDER BY similarity DESC
        LIMIT $k
        WITH COLLECT(DISTINCT n) AS targetNodes
        UNWIND targetNodes AS node
        MATCH (node)-[r]->(neighbor)
        RETURN node, r, neighbor"""
        for attempt in range(max_retries):
            try:
                async with self._driver.session() as session:
                    result = await session.run(
                        query=query, name_embedding=name_embedding, k=top_k
                    )
                    records = []
                    async for record in result:
                        records.append((record["node"], record["r"], record["neighbor"]))
                    return records
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                else:
                    return e