# src/chatbot.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import logging
from src.search.section_coarse_search import coarse_search_sections
from src.search.fine_search import fine_search_chunks
from src.inference.embedding_model import EmbeddingModel
from src.inference.llm_model import local_llm  # Example implementation of a local LLM
from src.utils.exceptions import SearchError, EmbeddingError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Answer the user's question based on the information provided in the document context below.\n"
    "Your response should reference the context clearly, but you may paraphrase or summarize appropriately."
)


class PDFChatBot:
    def __init__(self, sections, chunk_index, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        """
        Parameters
        ----------
        sections : list[dict]
            Each element contains keys such as ``"title"``, ``"title_emb"``,
            ``"avg_chunk_emb"``, etc.
        chunk_index : list[dict]
            Each chunk is a dictionary like
            ``{"embedding": [...], "metadata": {...}}``.
        system_prompt : str
            System prompt that is prepended before calling the LLM.
        """
        self.sections = sections
        self.chunk_index = chunk_index
        self.system_prompt = system_prompt

    def build_prompt(self, user_query, retrieved_chunks):
        """
        Construct the prompt that will be sent to the LLM.

        Parameters
        ----------
        user_query : str
            The question entered by the user.
        retrieved_chunks : list[dict]
            A list of chunks in the form
            ``[{"embedding": [...], "metadata": {...}}, ...]``.

        Returns
        -------
        str
            A fully formatted prompt string.
        """
        context_parts = []
        for item in retrieved_chunks:
            meta = item.get("metadata", {})
            section_title = meta.get("section_title", "")
            content = meta.get("content", "")
            context_parts.append(f"[{section_title}] {content}")
        context_text = "\n\n".join(context_parts)
        prompt = f"{self.system_prompt}\n\n=== Document Context ===\n{context_text}\n\n=== User Question ===\n{user_query}\n\n=== Answer ===\n"
        return prompt.strip()

    def answer(self, query: str, beta: float = 0.3, top_sections: int = 10, top_chunks: int = 5, streaming=False, fine_only=False):
        """
        End‑to‑end answer generation pipeline.

        Steps
        -----
        1. **Coarse Search** – Find the top *top_sections* sections at the section level.  
        2. **Fine Search** – Within those sections, retrieve the top *top_chunks* chunks.  
        3. **LLM Generation** – Send a prompt to the LLM and return the generated answer.

        Parameters
        ----------
        query : str
            User question.
        beta : float, default = 0.3
            Interpolation weight for coarse search scoring.
        top_sections : int, default = 10
            Number of sections to retain in the coarse search.
        top_chunks : int, default = 5
            Number of chunks to use in the fine search.
        streaming : bool, default = False
            If ``True``, stream tokens as they are generated.

        Returns
        -------
        str
            The LLM’s answer text.
        """
        chunk_index = self.chunk_index
        sections = self.sections
        if fine_only:              
            relevant_secs = self.sections
        else:
            # Coarse Search (section level)
            relevant_secs = coarse_search_sections(query, sections, beta=beta, top_k=top_sections)        
        # Fine Search (chunk level)
        try:
            query_emb = EmbeddingModel.get_embedding(query)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise EmbeddingError(f"Failed to generate query embedding: {e}")

        try:
            best_chunks = fine_search_chunks(query_emb, chunk_index, relevant_secs, top_k=top_chunks, fine_only=fine_only)
        except Exception as e:
            logger.error(f"Error in fine search: {e}")
            raise SearchError(f"Failed to perform fine search: {e}")

        # Build a single string that contains the content of every retrieved chunk
        combined_answer = "\n\n".join(
            chunk["metadata"].get("content", "") for chunk in best_chunks
        )

        # Ask the LLM to improve the user query based on ALL retrieved evidence
        query_improvement_prompt = (
            "The user question is: " + query + "\n\n"
            "The retrieved chunks are:\n" + combined_answer + "\n\n"
            "Based on the retrieved chunks above, generate a supplemental question that would help retrieve even more relevant information.\n Only display the final question.\n"
            "The improved question is: "
        )
        try:
            improved_query = local_llm.generate(query_improvement_prompt, streaming=streaming)
            # Safely extract improved query
            if "<|im_start|>assistant" in improved_query and "<|im_end|>" in improved_query:
                improved_query = improved_query.split("<|im_start|>assistant")[1].split("<|im_end|>")[0].strip()
            else:
                logger.warning("Could not parse improved query, using original query")
                improved_query = query
        except Exception as e:
            logger.error(f"Error generating improved query: {e}")
            improved_query = query
        if fine_only:              
            relevant_secs = self.sections
        else:
            # Coarse Search (section level)
            relevant_secs = coarse_search_sections(query + ':' + improved_query, sections, beta=beta, top_k=top_sections)
            
        # Fine Search (chunk level)
        try:
            query_emb = EmbeddingModel.get_embedding(query + ':' + improved_query)
        except Exception as e:
            logger.error(f"Error generating combined query embedding: {e}")
            raise EmbeddingError(f"Failed to generate combined query embedding: {e}")
        
        try:
            best_chunks = fine_search_chunks(query_emb, chunk_index, 
                                           relevant_secs, top_k=top_chunks, 
                                           fine_only=fine_only)
        except Exception as e:
            logger.error(f"Error in fine search with improved query: {e}")
            raise SearchError(f"Failed to perform fine search with improved query: {e}")
            
        # Generate LLM answer
        try:
            prompt = self.build_prompt(query, best_chunks)
            answer_text = local_llm.generate(prompt, streaming=streaming)
            return answer_text
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Sorry, an error occurred while generating the answer: {str(e)}"

if __name__ == "__main__":
    sections_path = "data/extracted/sections_with_emb.json"
    chunk_index_path = "data/index/sample_chunks_vectors.json"

    if not os.path.exists(sections_path):
        print(f"[ERROR] Sections file not found: {sections_path}")
        exit(1)
    else:
        with open(sections_path, 'r', encoding='utf-8') as f:
            sections = json.load(f)

    if not os.path.exists(chunk_index_path):
        print(f"[ERROR] Chunk index file not found: {chunk_index_path}")
        exit(1)
    else:
        with open(chunk_index_path, 'r', encoding='utf-8') as f:
            chunk_index = json.load(f)

    chatbot = PDFChatBot(sections, chunk_index)
    print("Chatbot is ready. Enter your question below:")

    while True:
        query = input("Question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = chatbot.answer(query, streaming=True)