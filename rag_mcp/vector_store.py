"""Persistent FAISS vector store shared by Flask and the MCP server."""

import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np

from config.settings import INDEX_DIR
from rag_mcp.embeddings import DIM, embed

log = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, folder=INDEX_DIR):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

        self.index_path = self.folder / "index.faiss"
        self.meta_path = self.folder / "meta.json"

    def _empty_index(self):
        return faiss.IndexFlatIP(DIM)

    def _load(self):
        """Load and validate the FAISS index and its matching metadata."""

        has_index = self.index_path.exists()
        has_meta = self.meta_path.exists()

        if not has_index and not has_meta:
            return self._empty_index(), []

        if has_index != has_meta:
            raise RuntimeError(
                "FAISS storage is incomplete. Both index.faiss and "
                "meta.json must exist. Restore a matching backup."
            )

        try:
            index = faiss.read_index(str(self.index_path))
            meta = json.loads(
                self.meta_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load FAISS storage: {exc}"
            ) from exc

        if index.d != DIM:
            raise RuntimeError(
                f"FAISS dimension mismatch: index={index.d}, "
                f"embedding={DIM}."
            )

        if not isinstance(meta, list):
            raise RuntimeError("FAISS metadata is not a list.")

        if index.ntotal != len(meta):
            raise RuntimeError(
                f"FAISS record mismatch: vectors={index.ntotal}, "
                f"metadata={len(meta)}. Restore a matching backup."
            )

        for item in meta:
            if (
                not isinstance(item, dict)
                or "source" not in item
                or "text" not in item
            ):
                raise RuntimeError("Invalid FAISS metadata entry.")

        return index, meta

    def _save(self, index, meta):
        """Write new files before replacing the existing files."""

        if index.ntotal != len(meta):
            raise RuntimeError(
                "Cannot save FAISS: vector and metadata counts differ."
            )

        index_tmp = self.folder / "index.faiss.tmp"
        meta_tmp = self.folder / "meta.json.tmp"

        try:
            faiss.write_index(index, str(index_tmp))

            meta_tmp.write_text(
                json.dumps(meta, ensure_ascii=False),
                encoding="utf-8",
            )

            os.replace(index_tmp, self.index_path)
            os.replace(meta_tmp, self.meta_path)

        finally:
            if index_tmp.exists():
                index_tmp.unlink()

            if meta_tmp.exists():
                meta_tmp.unlink()

    def add(self, source: str, chunks: list[str]) -> int:
        """Add chunks, replacing any previous chunks from the same source."""

        if not source:
            raise ValueError("Source cannot be empty.")

        chunks = [
            str(chunk).strip()
            for chunk in chunks
            if chunk and str(chunk).strip()
        ]

        index, meta = self._load()

        # Keep vectors belonging to all other documents.
        keep = [
            i for i, item in enumerate(meta)
            if item["source"] != source
        ]

        if index.ntotal:
            old_vectors = index.reconstruct_n(0, index.ntotal)
            old_vectors = old_vectors[keep]
        else:
            old_vectors = np.empty((0, DIM), dtype=np.float32)

        if chunks:
            new_vectors = np.asarray(
                embed(chunks),
                dtype=np.float32,
            )

            if new_vectors.shape != (len(chunks), DIM):
                raise RuntimeError(
                    f"Unexpected embedding shape: {new_vectors.shape}; "
                    f"expected {(len(chunks), DIM)}."
                )
        else:
            new_vectors = np.empty((0, DIM), dtype=np.float32)

        vectors = np.ascontiguousarray(
            np.vstack((old_vectors, new_vectors)),
            dtype=np.float32,
        )

        new_meta = (
            [meta[i] for i in keep]
            + [
                {"source": source, "text": chunk}
                for chunk in chunks
            ]
        )

        new_index = self._empty_index()

        if vectors.shape[0] > 0:
            new_index.add(vectors)

        self._save(new_index, new_meta)

        log.info(
            "FAISS updated: source=%s, added=%d, total=%d",
            source,
            len(chunks),
            new_index.ntotal,
        )

        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the most similar document chunks."""

        query = (query or "").strip()
        if not query:
            return []

        top_k = max(1, min(int(top_k), 10))

        index, meta = self._load()

        if index.ntotal == 0:
            return []

        query_vector = np.asarray(
            embed([query]),
            dtype=np.float32,
        )

        if query_vector.shape != (1, DIM):
            raise RuntimeError(
                f"Invalid query embedding shape: {query_vector.shape}."
            )

        query_vector = np.ascontiguousarray(query_vector)

        scores, ids = index.search(
            query_vector,
            min(top_k, index.ntotal),
        )

        results = []

        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue

            item = meta[int(idx)]

            results.append({
                "source": item["source"],
                "score": round(float(score), 4),
                "text": item["text"],
            })

        return results

    def sources(self) -> list[dict]:
        """Return indexed document names and chunk counts."""

        _, meta = self._load()

        counts = {}

        for item in meta:
            source = item["source"]
            counts[source] = counts.get(source, 0) + 1

        return [
            {"source": source, "chunks": count}
            for source, count in counts.items()
        ]