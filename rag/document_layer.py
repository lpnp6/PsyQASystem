from rag.data import Node, Document
from rag.base import GraphDatabaseHandler
from rag.utils import logger
import uuid
import time
import asyncio


async def chunk_by_sematic(
    document: Document, embedding, chunk_size=800, overlap_size=200
) -> list[Node]:
    def _create_chunk(overlap, current):
        return " ".join([s.text for s in overlap + current])

    def _get_overlap(sentences, overlap_tokens):
        overlap = []
        total = 0
        for sent in reversed(sentences):
            if total + len(sent) > overlap_tokens and total > 0:
                break
            overlap.insert(0, sent)
            total += len(sent)
        return overlap

    sents = await embedding.get_sents(document.text)
    chunks = []
    current_chunk = []
    overlap_buffer = []
    current_length = 0

    for sent in sents:
        sent_len = len(sent.text)

        if len(sent.text) > chunk_size:
            if current_chunk:
                chunks.append(_create_chunk(overlap_buffer, current_chunk))
                overlap_buffer = _get_overlap(current_chunk, overlap_size)
                current_chunk = []
                current_length = 0
            chunks.append(sent.text)
            overlap_buffer = []
            continue

        total_length = sum(len(s) for s in overlap_buffer) + current_length + sent_len

        if total_length > chunk_size:
            if current_chunk:
                chunks.append(_create_chunk(overlap_buffer, current_chunk))
                overlap_buffer = _get_overlap(current_chunk, overlap_size)
                current_chunk = [sent]
                current_length = sent_len
            else:
                chunks.append(_create_chunk(overlap_buffer, [sent]))
                overlap_buffer = _get_overlap([sent], overlap_size)
        else:
            current_chunk.append(sent)
            current_length += sent_len

    if current_chunk:
        chunks.append(_create_chunk(overlap_buffer, current_chunk))
    nodes = []
    offset = 0
    tasks = [embedding.get_embeddings(chunk) for chunk in chunks]
    embeddings = await asyncio.gather(*tasks)
    document_id = document.id
    for i, chunk in enumerate(chunks):
        nodes.append(
            Node(
                label="Chunk",
                properties={
                    "node_id": "node-" + str(uuid.uuid4()),
                    "document_id": document_id,
                    "document_title": document.title,
                    "offset": offset,
                    "tokens": len(chunk),
                    "update_time": document.update_time,
                    "text": chunk,
                    "embedding": embeddings[i],
                },
            )
        )
        offset += len(chunk)
    return nodes


async def insert_chunk(chunk_node: Node, handler: GraphDatabaseHandler):
    try:
        node = await handler.insert_node(chunk_node)
        return node
    except Exception as e:
        logger.error(f"exception occured at insert_chunk: {e}")
