from rag.base import GraphDatabaseHandler
from rag.embedding import Embedding_model
from rag.data import Node
import asyncio


def cal_similarity(query: str, answer):
    pass


async def extract_subject_predicate(
    embedding: Embedding_model, text: str
) -> tuple[str, str]:
    doc = embedding.model(text)
    subject_span = None
    predicate_span = None

    for token in doc:
        if token.dep_ in ("nsubj", "nsubj:pass"):
            subject_span = doc[token.left_edge.i : token.right_edge.i + 1]
            break

    root = next((token for token in doc if token.dep_ == "ROOT"), None)
    if root:
        predicate_tokens = [
            token
            for token in root.subtree
            if not (subject_span and token in subject_span)
        ]
        if predicate_tokens:
            start = predicate_tokens[0].i
            end = predicate_tokens[-1].i + 1
            predicate_span = doc[start:end]

    return subject_span, predicate_span


async def retrieve(
    query: str, embedding: Embedding_model, gragh_db_handler: GraphDatabaseHandler, threshold = 0.95
):
    subject_span, predicate_span = await extract_subject_predicate(
        embedding=embedding, text=query
    )

    subject_embedding = await embedding.get_embeddings(subject_span)
    node = Node(label="Entity", properties={"name": subject_embedding})
    ans = []
    if all(element == 0 for element in subject_embedding):
        results = await gragh_db_handler.keyword_search(node=node)
        for result in results:
            subject = result[0]
            predicate = result[1]
            object = result[2]
            
            if cal_similarity(predicate.properties.get("name"), predicate_span) >= threshold:
                ans.append([subject, predicate, object])
    else:
        results = await gragh_db_handler.semantic_search(node=node)
        for result in results:
            subject = result[0]
            predicate = result[1]
            object = result[2]
            
            if cal_similarity(subject.properties.get("name") + predicate.properties.get("name"), subject_span + predicate_span) >= threshold:
                ans.append([subject, predicate, object]) 