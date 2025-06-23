# src/utils/similarity.py

import numpy as np
from typing import List, Union

def cosine_similarity(vec1: Union[List[float], np.ndarray], 
                     vec2: Union[List[float], np.ndarray]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Parameters
    ----------
    vec1 : Union[List[float], np.ndarray]
        First vector
    vec2 : Union[List[float], np.ndarray]
        Second vector
        
    Returns
    -------
    float
        Cosine similarity score between -1 and 1
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)