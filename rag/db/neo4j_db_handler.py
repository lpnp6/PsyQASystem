from neo4j import AsyncGraphDatabase, GraphDatabase, exceptions
from dataclasses import dataclass
from rag.data import Node
from rag.utils import logger
from rag.base import GraphDatabaseHandler
import numpy as np
import asyncio
import uuid
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
        DATABASE = os.environ.get(
            "NEO4Jdatabase", re.sub(r"[^a-zA-Z0-9-]", "-", namespace)
        )
        self._driver = AsyncGraphDatabase.driver(
            URI,
            auth=(USERNAME, PASSWORD),
            max_connection_lifetime=30 * 60,  # 30分钟重新建立连接
            connection_timeout=60,  # 60秒连接超时
            keep_alive=True,  # 启用TCP keep-alive
            connection_acquisition_timeout=60,
            max_connection_pool_size=MAX_CONNECTION_POOL_SIZE,
        )
        self._driver_lock = asyncio.Lock()

        with GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD)) as _sync_driver:
            for database in (DATABASE, None):
                self._database = database
                connected = False
                try:
                    with _sync_driver.session(database=database) as session:
                        try:
                            session.run("MATCH (n) RETURN n LIMIT 0")
                            logger.info(f"Connected to {DATABASE} at{URI}")
                            connected = True
                        except exceptions.ServiceUnavailable as e:
                            logger.error(
                                f"{database} at {URI} is not available".capitalize()
                            )
                            raise e
                except exceptions.AuthError as e:
                    logger.error(f"Authentication failed for {database} as {URI}")
                    raise e
                except exceptions.ClientError as e:
                    if e.code == "Neo.ClientError.Database.DatabaseNotFound":
                        logger.info(
                            f"{database} not found. Try to creat a specialized database.".capitalize()
                        )
                        try:
                            with _sync_driver.session() as session:
                                session.run(
                                    f"CREATE DATABASE `{database}` IF NOT EXISTS"
                                )
                                logger.info(f"{database} at {URI} created".capitalize())
                                connected = True
                        except (exceptions.ClientError, exceptions.DatabaseError) as e:
                            if (
                                e.code
                                == "Neo.ClientError.Statement.UnsupportedAdministrationCommand"
                                or e.code
                                == "Neo.DatabaseError.Statement.ExecutionFailed"
                            ):
                                if database is not None:
                                    logger.warning(
                                        "This Neo4j instance does not support creating databases. Try to use Neo4j Desktop/Enterprise version or DozerDB instead. Fallback to use the default database."
                                    )
                            if database is None:
                                logger.error(f"Failed to create {database} at {URI}")
                                raise e
                except Exception as e:
                    pass
                if connected:
                    break

    def close(self):
        self.driver.close()

    async def insert_node(self, n: Node):
        label = n.label
        query = f"""
        MERGE (n:{label} {{node_id: $node_id}})
        on CREATE SET n = $properties
        on MATCH SET n += $properties
        return n
        """
        async with self._driver.session(database=self._database) as session:
            record = await session.run(
                query=query, node_id=n.properties["node_id"], properties=n.properties
            )
            node = await record.single()
            if node is not None:
                return n
            else:
                return None

    async def insert_edge(self, node1, node2, edge):
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
        async with self._driver.session() as session:
            result = await session.run(
                query=query,
                node_id1=node_id1,
                node_id2=node_id2,
                properties=properties,
            )
            return await result.single()