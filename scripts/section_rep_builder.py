# scripts/section_rep_builder.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from src.inference.embedding_model import EmbeddingModel

def build_section_reps(sections, chunk_index):
    """
    sections: [
      { "title": "Chapter 2 Installation", "start_page":10, "end_page":19, ... },
      ...
    ]
    chunk_index: [{ "embedding": [...], "metadata": {"section_title": "...", ...}}, ...]

    => Adds sec["title_emb"], sec["avg_chunk_emb"] fields to each section
    """
    # 1) Section title embeddings (batch)
    titles = [sec["title"] for sec in sections]
    embedder = EmbeddingModel()    
    title_embs  = embedder.get_embeddings(titles)# shape: (num_sections, dim)
    for i, sec in enumerate(sections):
        sec["title_emb"] = title_embs[i].tolist()

    # 2) Collect chunks by section
    section2embs = {}
    for item in chunk_index:
        sec_t = item["metadata"]["section_title"]
        emb = item["embedding"]  # list[float]
        if sec_t not in section2embs:
            section2embs[sec_t] = []
        section2embs[sec_t].append(emb)

    # 3) Average embeddings of chunks within each section
    for sec in sections:
        stitle = sec["title"]
        if stitle not in section2embs:
            sec["avg_chunk_emb"] = None
        else:
            arr = np.array(section2embs[stitle])  # shape: (num_chunks, emb_dim)
            avg_vec = arr.mean(axis=0)            # (emb_dim,)
            sec["avg_chunk_emb"] = avg_vec.tolist()
    
    return sections

if __name__ == "__main__":
    # Example: data/extracted/sections.json (TOC-based section info)
    sections_json = "data/extracted/sections.json"
    # Example: data/index/sample_chunks_vectors.json (chunk embeddings)
    chunk_index_json = "data/index/sample_chunks_vectors.json"

    with open(sections_json, 'r', encoding='utf-8') as f:
        sections_data = json.load(f)
    
    with open(chunk_index_json, 'r', encoding='utf-8') as f:
        chunk_index_data = json.load(f)

    # Generate section representative vectors
    updated_sections = build_section_reps(sections_data, chunk_index_data)

    # Save (e.g., data/extracted/sections_with_emb.json)
    out_path = "data/extracted/sections_with_emb.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(updated_sections, f, ensure_ascii=False, indent=2)

    print("Section reps built and saved.")