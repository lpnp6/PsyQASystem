from rag.data import Node, Edge
from rag.base import BaseLLM, GraphDatabaseHandler
from rag.embedding import Embedding_model
from rag.utils import UnionFind, logger
from collections import defaultdict
from tqdm.asyncio import tqdm_asyncio
from dataclasses import asdict
import numpy as np
import pandas as pd
import os
import faiss
import re
import regex
import uuid
import json
import asyncio


def build_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def query_candidates(query_vec, index, threshold=0.95) -> list[int]:
    query_vec = query_vec.reshape(1, -1)
    distances, indices = index.search(query_vec, k=5)
    return [idx for d, idx in zip(distances[0], indices[0]) if d >= threshold]


def is_entity_duplicate(e1, e2, threshold=0.95):
    sim_name = np.dot(e1["name_embedding"], e2["name_embedding"])
    sim_type = np.dot(e1["type_embedding"], e2["type_embedding"])
    sim_description = np.dot(e1["description_embedding"], e2["description_embedding"])
    sim = sim_name * 0.6 + sim_type * 0.3 + sim_description * 0.1
    if sim >= threshold:
        return True
    else:
        return False


def is_relation_duplicate(r1, r2, threshold=0.95):
    sim_name = np.dot(r1["name_embedding"], r2["name_embedding"])
    sim_description = np.dot(r1["description_embedding"], r2["description_embedding"])
    sim = sim_name * 0.7 + sim_description * 0.3
    if sim >= threshold:
        return True
    else:
        return False

async def build_llm_based(
    chunk_nodes: Node,
    graph_db_handler: GraphDatabaseHandler,
    llm_model: BaseLLM,
    embedding_model: Embedding_model,
):
    entities = []
    relations = []
    tasks = []
    for chunk_node in chunk_nodes:
        chunk_id = chunk_node.properties["node_id"]
        document_id = chunk_node.properties["document_id"]
        document_title = chunk_node.properties["document_title"]
        offset = chunk_node.properties["offset"]
        update_time = chunk_node.properties["update_time"]
        text = chunk_node.properties["text"]
        tasks.append(asyncio.create_task(llm_model.generate(text)))
    results = await tqdm_asyncio.gather(*tasks)
    for result in results:
        if isinstance(result, Exception):
            continue
        try:
            js = json.loads(result.strip("```").strip("json"))
        except json.JSONDecodeError as e:
            logger.error(e)
            continue
        entities.extend(js["entities"])
        relations.extend(js["relations"])

    results = dict()
    tasks = []
    node_df = pd.DataFrame(
        columns=[
            "node_id",
            "chunk_id",
            "document_id",
            "document_title",
            "type",
            "name",
            "description",
            "title_embedding",
            "type_embedding",
            "name_embedding",
            "description_embedding",
            "update_time",
        ]
    )

    name_embeddings = None
    for entity in entities:

        node_id = "node-" + str(uuid.uuid4())
        type = entity["type"]
        name = entity["name"]
        description = entity["description"]
        title_embedding = await embedding_model.get_embeddings(document_title)
        type_embedding = await embedding_model.get_embeddings(type)
        name_embedding = await embedding_model.get_embeddings(name)
        description_embedding = await embedding_model.get_embeddings(description)
        node_df.loc[len(node_df)] = [
            node_id,
            chunk_id,
            document_id,
            document_title,
            type,
            name,
            description,
            title_embedding,
            type_embedding,
            name_embedding,
            description_embedding,
            update_time,
        ]
        if name_embeddings is not None and not all(
            element == 0 for element in name_embedding
        ):
            name_embeddings = np.vstack((name_embeddings, np.array(name_embedding)))
        elif not all(element == 0 for element in name_embedding):
            name_embeddings = np.array([name_embedding])
    name_index = build_index(name_embeddings)

    node_uf = UnionFind(len(entities))
    for i in range(len(entities)):
        candidates = query_candidates(node_df.loc[i]["name_embedding"], name_index)
        for j in candidates:
            if i < j and is_entity_duplicate(
                node_df.loc[i].to_dict(), node_df.loc[j].to_dict()
            ):
                node_uf.union(i, j)

    nodes_harsh = dict()
    entity_clusters = defaultdict(list)
    for i in range(len(entities)):
        type = entities[i]["type"]
        name = entities[i]["name"]
        nodes_harsh[type + name] = node_uf.find(i)
        entity_clusters[node_uf.find(i)].append(i)

    emerged_nodes = dict()
    for k, v in entity_clusters.items():
        cluster = [node_df.loc[i].to_dict() for i in v]
        node_id = min([entity["node_id"] for entity in cluster])
        type = entities[k]["type"]
        name = entities[k]["name"]
        description = entities[k]["description"]
        type_embedding = np.mean(
            np.array([entity["type_embedding"] for entity in cluster]), axis=0
        )
        name_embedding = np.mean(
            np.array([entity["name_embedding"] for entity in cluster]), axis=0
        )
        description_embedding = np.mean(
            np.array([entity["description_embedding"] for entity in cluster]), axis=0
        )
        entity_node = Node(
            label="Entity",
            properties={
                "node_id": node_id,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "document_title": document_title,
                "type": type,
                "name": name,
                "description": description,
                "title_embedding": title_embedding,
                "type_embedding": type_embedding,
                "name_embedding": name_embedding,
                "description_embedding": description_embedding,
                "update_time": update_time,
            },
        )
        emerged_nodes[k] = entity_node
        tasks.append(asyncio.create_task(graph_db_handler.insert_node(entity_node)))
    results["entities"] = await asyncio.gather(*tasks)
    tasks = []
    for node in emerged_nodes.values():
        tasks.append(asyncio.create_task(graph_db_handler.merge(node=node)))
    await tqdm_asyncio.gather(*tasks)
    edge_df = pd.DataFrame(
        columns=[
            "edge_id",
            "name",
            "description",
            "name_embedding",
            "description_embedding",
            "from",
            "to",
            "update_time",
        ]
    )

    name_embeddings = None
    description_embeddings = None

    tasks = []
    for relation in relations:
        entity1 = relation["entity1"]
        cluster_key1 = nodes_harsh.get(entity1["type"] + entity1["name"])
        if cluster_key1 is None:
            continue
        node1 = emerged_nodes[cluster_key1]
        entity2 = relation["entity2"]
        cluster_key2 = nodes_harsh.get(entity2["type"] + entity2["name"])
        if cluster_key2 is None:
            continue
        node2 = emerged_nodes[cluster_key2]
        edge_id = "edge-" + str(uuid.uuid4())
        relation_name = relation["relation"]
        description = relation["description"]
        name_embedding = await embedding_model.get_embeddings(relation_name)
        description_embedding = await embedding_model.get_embeddings(description)
        edge_df.loc[len(edge_df)] = [
            edge_id,
            relation_name,
            description,
            name_embedding,
            description_embedding,
            node1,
            node2,
            update_time,
        ]
        if name_embeddings is not None and not all(
            element == 0 for element in name_embedding
        ):
            name_embeddings = np.vstack((name_embeddings, np.array(name_embedding)))
        elif not all(element == 0 for element in name_embedding):
            name_embeddings = np.array(name_embedding)

    name_index = build_index(name_embeddings)

    edge_uf = UnionFind(len(edge_df))
    for i in range(len(edge_df)):
        candidates = query_candidates(edge_df.loc[i]["name_embedding"], name_index)
        for j in candidates:
            if (
                i < j
                and edge_df.loc[i]["from"] is edge_df.loc[j]["from"]
                and edge_df.loc[i]["to"] is edge_df.loc[j]["to"]
                and is_relation_duplicate(
                    edge_df.loc[i].to_dict(), edge_df.loc[j].to_dict()
                )
            ):
                edge_uf.union(i, j)

    edge_clusters = defaultdict(list)
    for i in range(len(edge_df)):
        name = relations[i]["relation"]
        description = relations[i]["description"]
        edge_clusters[edge_uf.find(i)].append(i)

    for k, v in edge_clusters.items():
        edge_id = min([edge_df.loc[i]["edge_id"] for i in v])
        name = edge_df.loc[k]["name"]
        description = edge_df.loc[k]["description"]
        name_embedding = np.mean(
            np.array([edge_df.loc[i]["name_embedding"] for i in v]), axis=0
        )
        description_embedding = np.mean(
            np.array([edge_df.loc[i]["description_embedding"] for i in v]), axis=0
        )
        from_node = edge_df.loc[k]["from"]
        to_node = edge_df.loc[k]["to"]
        relation_edge = Edge(
            label="Relation",
            properties={
                "edge_id": edge_id,
                "name": name,
                "description": description,
                "name_embedding": name_embedding,
                "description_embedding": description_embedding,
                "from": from_node.properties["node_id"],
                "to": to_node.properties["node_id"],
                "update_time": update_time,
            },
        )
        tasks.append(
            asyncio.create_task(
                graph_db_handler.insert_edge(
                    node1=from_node, node2=to_node, edge=relation_edge
                )
            )
        )
    results["relations"] = await asyncio.gather(*tasks)
    return results



