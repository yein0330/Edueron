# src/search/section_coarse_search.py

import numpy as np
from ..inference.embedding_model import embedding_model
from src.utils.similarity import cosine_similarity

def coarse_search_sections(query: str, sections: list, beta=0.3, top_k=5):
    """
    Select the most relevant document sections for the given query using a
    two‑stage cosine‑similarity score.

    Parameters
    ----------
    query : str
        User query text.
    sections : list[dict]
        Each element must contain:
        {
            "title": str,
            "title_emb": list[float],
            "avg_chunk_emb": list[float],
            ...
        }
    beta : float, default = 0.3
        Interpolation weight between title similarity and average‑chunk similarity.
    top_k : int, default = 5
        Number of top‑scoring sections to return.

    Notes
    -----
    Cosine similarity is computed between the query embedding and:
    1. the section title embedding (``title_emb``), and
    2. the section’s average chunk embedding (``avg_chunk_emb``).

    The final score is calculated as::

        final_score = beta * sim_title + (1 - beta) * sim_chunk
    """
    query_emb = embedding_model.get_embedding(query)

    scored = []
    for sec in sections:
        title_emb = sec.get("title_emb")
        chunk_emb = sec.get("avg_chunk_emb")
        if title_emb is None or chunk_emb is None:
            # Skip if embeddings are missing
            continue
        sim_title = cosine_similarity(query_emb, title_emb)
        sim_chunk = cosine_similarity(query_emb, chunk_emb)

        final_score = beta * sim_title + (1 - beta) * sim_chunk
        scored.append((final_score, sec))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top_sections = [x[1] for x in scored[:top_k]]
    return top_sections