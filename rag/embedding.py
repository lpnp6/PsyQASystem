from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from rag.utils import logger
import asyncio
import spacy
import threading
import numpy as np


@dataclass
class Embedding_model:
    embedding_dim: int = 300
    max_token_size: int = 512
    lock: threading.Lock = field(default_factory=threading.Lock)

    async def get_embeddings(self, text):
        raise NotImplementedError

    async def get_sents(self, text):
        raise NotImplementedError

    async def extract_key_words(self, text):
        raise NotImplementedError


@dataclass
class Spacy(Embedding_model):
    model_name: str = field(default="zh_core_web_trf")
    embedding_dim: int = field(default=768)
    _model: Optional[Any] = field(init=False, default=None)
    _executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1024)
    )

    @property
    def model(self):
        if self._model is None:
            self._model = spacy.load(self.model_name)
        return self._model

    def load_model(self, mode_name: str) -> None:
        with self.lock:
            self._model = spacy.load(mode_name)
            self.model_name = mode_name

    def _sync_get_embeddings(self, text):
        with self.lock:
            return Spacy.normalize(self.model(text).vector)

    async def get_embeddings(self, text):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._sync_get_embeddings, text
        )

    def _sync_get_sents(self, text):
        with self.lock:
            return list(self.model(text).sents)

    async def get_sents(self, text):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_get_sents, text)

    def normalize(embeddings):
        embeddings = np.array(embeddings, dtype=np.float32)
        axis = 1 if embeddings.ndim == 2 else 0
        norms = np.linalg.norm(embeddings, axis=axis, keepdims=True)
        norms[norms == 0] = 1e-10
        return embeddings / norms
    
if __name__ == "__main__":
     m = Spacy()   
     doc = m.model("公司位于？")
     print("ok")
