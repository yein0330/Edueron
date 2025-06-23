# src/search/fine_search.py

import numpy as np

def fine_search_chunks(query_emb, chunk_index, target_sections, top_k=10, fine_only=False):
    """
    Find the most relevant text chunks within the specified sections.

    Parameters
    ----------
    query_emb : list[float] | np.ndarray
        Embedding vector of the user query.
    chunk_index : list[dict]
        Each element is a dictionary like:
        {
            "embedding": [...],
            "metadata": {"section_title": "...", ...}
        }
    target_sections : list[dict]
        Sections to search within, e.g.,
        [
            {"title": "Section 2 Installation Guide", ...},
            ...
        ]
    top_k : int, default = 10
        Number of top‑scoring chunks to return.

    Notes
    -----
    - Only chunks whose ``section_title`` appears in *target_sections* are considered.
    - Cosine similarity is computed between the query embedding and each candidate
      chunk. The chunks are then sorted in descending order of similarity and the
      top *k* results are returned.
    """

    section_titles = [sec["title"] for sec in target_sections]
    candidates = chunk_index
    if not fine_only:
        candidates = [
            item for item in candidates
            if item["metadata"]["section_title"] in section_titles
        ]
        if len(candidates) == 0:
            candidates = chunk_index
    results = []
    qv = np.array(query_emb)
    q_norm = np.linalg.norm(qv)
    for c in candidates:
        emb = np.array(c["embedding"])
        dot = np.dot(qv, emb)
        denom = np.linalg.norm(emb) * q_norm + 1e-8
        cos_val = dot / denom
        results.append((cos_val, c))

    results.sort(key=lambda x: x[0], reverse=True)

    top_results = [r[1] for r in results[:top_k]]
    return top_results
    