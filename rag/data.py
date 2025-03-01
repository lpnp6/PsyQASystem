from dataclasses import dataclass, field
import time
import hashlib


@dataclass
class Node:
    label: str
    properties: dict


@dataclass
class Document:
    id: str
    title: str
    text: str
    token: int
    update_time: float = field(default=time.time())


@dataclass
class Edge:
    label: str
    properties: dict = field(default_factory=dict)


@dataclass
class Index:
    _index: str
    vector: list
    node_id: str
    properties: dict

    def __hash__(self):
        hash_object =  hashlib.sha256(self.node_id.encode())
        return int(hash_object.hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, Index):
            return self.node_id == other.node_id
        return False
