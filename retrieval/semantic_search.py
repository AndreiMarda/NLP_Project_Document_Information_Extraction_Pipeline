# retrieval/semantic_search.py

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from nlp.segmentation import segment_paragraphs


@dataclass(frozen=True)
class IndexedChunk:
    doc_id: str
    paragraph_id: int
    text: str


class SemanticCorpusIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.nn: Optional[NearestNeighbors] = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[IndexedChunk] = []

    def build_from_docs(self, docs, min_par_len: int = 50) -> None:
        """
        docs: List[LoadedDoc]
        Creates paragraph chunks across all docs and embeds them.
        """
        chunks: List[IndexedChunk] = []
        texts: List[str] = []

        for d in docs:
            paras = segment_paragraphs(d.text, min_len=min_par_len)
            for p in paras:
                chunks.append(IndexedChunk(doc_id=d.doc_id, paragraph_id=p.paragraph_id, text=p.text))
                texts.append(p.text)

        if not texts:
            raise ValueError("No paragraphs found to index (try lowering min_par_len).")

        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        emb = np.asarray(emb, dtype=np.float32)

        self.chunks = chunks
        self.embeddings = emb

        # cosine distance = 1 - cosine similarity (works with normalized vectors)
        self.nn = NearestNeighbors(metric="cosine", algorithm="auto")
        self.nn.fit(self.embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, IndexedChunk]]:
        if self.nn is None or self.embeddings is None:
            raise RuntimeError("Index not built. Call build_from_docs() or load().")

        q_emb = self.model.encode([query], normalize_embeddings=True)
        distances, idxs = self.nn.kneighbors(q_emb, n_neighbors=min(top_k, len(self.chunks)))
        distances = distances[0]
        idxs = idxs[0]

        results = []
        for dist, i in zip(distances, idxs):
            # cosine similarity = 1 - cosine distance
            score = float(1.0 - dist)
            results.append((score, self.chunks[int(i)]))
        return results

    def save(self, path: str) -> None:
        if self.nn is None or self.embeddings is None:
            raise RuntimeError("Nothing to save. Build the index first.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "embeddings": self.embeddings,
                },
                f,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]

        self.nn = NearestNeighbors(metric="cosine", algorithm="auto")
        self.nn.fit(self.embeddings)
