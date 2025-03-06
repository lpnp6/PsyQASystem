from rag.base import GraphDatabaseHandler
from rag.embedding import Embedding_model
import asyncio


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
    query: str, embedding: Embedding_model, gragh_db_handler: GraphDatabaseHandler
):
    subject_span, predicate_span = await extract_subject_predicate(
        embedding=embedding, text=query
    )
    
if __name__ == "__main__":
    print(asyncio
          .run(extract_subject_predicate("伊安·麦克德摩写了哪本书")))