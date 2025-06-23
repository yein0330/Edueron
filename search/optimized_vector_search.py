# src/search/optimized_vector_search.py

import numpy as np
import torch
from typing import List, Dict, Union, Tuple
from src.utils.device import get_device_and_attention

class OptimizedVectorSearch:
    """
    Optimized vector search using PyTorch for batch operations.
    Falls back to NumPy if CUDA/MPS is not available.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the optimized vector search.
        
        Parameters
        ----------
        use_gpu : bool
            Whether to use GPU acceleration if available
        """
        self.device, _ = get_device_and_attention()
        self.use_gpu = use_gpu and self.device != "cpu"
        
    def batch_cosine_similarity(self, 
                               query_emb: Union[np.ndarray, List[float]], 
                               embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and multiple embeddings in batch.
        
        Parameters
        ----------
        query_emb : Union[np.ndarray, List[float]]
            Query embedding vector
        embeddings : np.ndarray
            Matrix of embeddings (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Similarity scores for each embedding
        """
        if self.use_gpu and torch.cuda.is_available():
            return self._torch_batch_similarity(query_emb, embeddings)
        else:
            return self._numpy_batch_similarity(query_emb, embeddings)
    
    def _torch_batch_similarity(self, query_emb: Union[np.ndarray, List[float]], 
                               embeddings: np.ndarray) -> np.ndarray:
        """GPU-accelerated batch cosine similarity using PyTorch."""
        # Convert to tensors
        query_tensor = torch.tensor(query_emb, dtype=torch.float32, device=self.device)
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        
        # Normalize vectors
        query_norm = query_tensor / (torch.norm(query_tensor) + 1e-8)
        embeddings_norm = embeddings_tensor / (torch.norm(embeddings_tensor, dim=1, keepdim=True) + 1e-8)
        
        # Compute similarities
        similarities = torch.matmul(embeddings_norm, query_norm)
        
        return similarities.cpu().numpy()
    
    def _numpy_batch_similarity(self, query_emb: Union[np.ndarray, List[float]], 
                               embeddings: np.ndarray) -> np.ndarray:
        """Optimized NumPy batch cosine similarity."""
        query_vec = np.array(query_emb, dtype=np.float32)
        
        # Normalize query vector
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        # Normalize all embeddings at once
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute all similarities at once
        similarities = np.dot(embeddings_norm, query_norm)
        
        return similarities
    
    def search(self, query_emb: Union[np.ndarray, List[float]], 
               index_data: List[Dict], 
               top_k: int = 8,
               batch_size: int = 1000) -> List[Dict]:
        """
        Perform optimized vector search.
        
        Parameters
        ----------
        query_emb : Union[np.ndarray, List[float]]
            Query embedding
        index_data : List[Dict]
            List of items with 'embedding' and 'metadata'
        top_k : int
            Number of top results to return
        batch_size : int
            Batch size for processing large datasets
            
        Returns
        -------
        List[Dict]
            Top k most similar items
        """
        n_items = len(index_data)
        all_scores = np.zeros(n_items, dtype=np.float32)
        
        # Process in batches for memory efficiency
        for i in range(0, n_items, batch_size):
            batch_end = min(i + batch_size, n_items)
            batch_items = index_data[i:batch_end]
            
            # Extract embeddings for this batch
            batch_embeddings = np.array([item["embedding"] for item in batch_items], dtype=np.float32)
            
            # Compute similarities
            batch_scores = self.batch_cosine_similarity(query_emb, batch_embeddings)
            all_scores[i:batch_end] = batch_scores
        
        # Get top k indices
        if top_k >= n_items:
            top_indices = np.argsort(all_scores)[::-1]
        else:
            # Use argpartition for better performance with large arrays
            top_indices = np.argpartition(all_scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(all_scores[top_indices])[::-1]]
        
        # Return top k items
        return [index_data[idx] for idx in top_indices]

# Global instance
_search_engine = None

def get_optimized_search_engine() -> OptimizedVectorSearch:
    """Get or create the global optimized search engine."""
    global _search_engine
    if _search_engine is None:
        _search_engine = OptimizedVectorSearch()
    return _search_engine