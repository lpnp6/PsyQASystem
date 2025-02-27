from elasticsearch import AsyncElasticsearch, Elasticsearch, helpers
from dataclasses import dataclass, asdict
from rag.base import VectorDatabaseHandler
from rag.utils import logger
from rag.data import Index
import os


@dataclass
class ElasticsearchDatabaseHandler(VectorDatabaseHandler):
    def __init__(self, namespace, global_config):
        super().__init__(namespace=namespace, global_config=global_config)
        self.url = os.environ.get("ELASTICSEARCH_URL")
        self.__apikey = os.environ.get("ELASTICSEARCH_APIKEY")
        self.high_level_index_name = os.environ.get("HIGH_LEVEL_INDEX_NAME")
        self.low_level_index_name = os.environ.get("LOW_LEVEL_INDEX_NAME")
        self.client = AsyncElasticsearch(
            hosts=[self.url], api_key=self.__apikey, retry_on_timeout=True
        )
        self.mapping = self.global_config.get("vector_database_mapping")
        self.top_k = self.global_config.get("top_k")
        with Elasticsearch(hosts=[self.url], api_key=self.__apikey) as _sync_client:
            if not _sync_client.indices.exists(index=self.high_level_index_name):
                try:
                    self._sync_client.indices.put_mapping(
                        body=self.mapping, index=self.high_level_index_name
                    )
                    logger.info(f"Create index {self.high_level_index_name}")
                except Exception as e:
                    logger.error(
                        f"Failed to create high level index {self.high_level_index_name}. {e}"
                    )
                    raise e
            if not _sync_client.indices.exists(index=self.low_level_index_name):
                try:
                    self._sync_client.indices.put_mapping(
                        body=self.mapping, index=self.low_level_index_name
                    )
                    logger.info(f"Create index {self.low_level_index_name}")
                except Exception as e:
                    logger.error(
                        f"Failed to create high level index {self.low_level_index_name}. {e}"
                    )
                    raise e

        logger.info(f"Connected to vector database at {self.url}")

    async def insert_index(self, indices: list[Index]):
        bulk = [asdict(index) for index in indices]
        results = await helpers.async_bulk(self.client, bulk)
        return results

    async def get_index(self, index: Index):
        query_vector = index.vector
        query_body = {
            "size": self.top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": query_vector},
                    },
                }
            },
        }
        response = await self.client.search(index=index._index, body=query_body)
        results = []
        for hit in response["hits"]["hits"]:
            results.append(
                Index(
                    _index=index._index,
                    vector=hit["_source"]["vector"],
                    node_id=hit["_source"]["node_id"],
                    properties=hit["_source"]["properties"],
                )
            )
        return results
