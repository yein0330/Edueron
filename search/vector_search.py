# src/search/vector_search.py

import numpy as np
from typing import List, Dict
from src.utils.similarity import cosine_similarity
from src.optimized_vector_search import get_optimized_search_engine

def simple_vector_search(query_emb, index_data: List[Dict], top_k=8):
    """
    Optimized vector search using batch operations.
    
    Parameters
    ----------
    query_emb : numpy array or list[float]
        Query embedding vector
    index_data : List[Dict]
        List of items with format [{"embedding": [...], "metadata": {...}}, ...]
    top_k : int
        Number of top results to return
        
    Returns
    -------
    List[Dict]
        Top k most similar items
    """
    # Use optimized search engine
    search_engine = get_optimized_search_engine()
    return search_engine.search(query_emb, index_data, top_k=top_k)