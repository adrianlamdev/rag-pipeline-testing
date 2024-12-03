from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from typing import List, Tuple, Dict
import numpy as np
import time

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
    "icl": {
        "query": "Convert this example into vector to look for useful examples: ",
        "key": "Convert this example into vector for retrieval: ",
    },
    "chat": {
        "query": "Embed this dialogue to find useful historical dialogues: ",
        "key": "Embed this historical dialogue for retrieval: ",
    },
    "lrlm": {
        "query": "Embed this text chunk for finding useful historical chunks: ",
        "key": "Embed this historical text chunk for retrieval: ",
    },
    "tool": {
        "query": "Transform this user request for fetching helpful tool descriptions: ",
        "key": "Transform this tool description for retrieval: ",
    },
    "convsearch": {
        "query": "Encode this query and context for searching relevant passages: ",
        "key": "Encode this passage for retrieval: ",
    },
}


class RAG:
    def __init__(
        self,
        base_model_name: str = "BAAI/bge-base-en-v1.5",
        query_model_name: str = "BAAI/llm-embedder",
        reranker_name="BAAI/bge-reranker-v2-m3",
    ):
        self.base_model = SentenceTransformer(base_model_name)
        self.query_model = SentenceTransformer(query_model_name)
        self.chunks: List[Dict] = []
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.reranker = CrossEncoder(reranker_name)

    def chunk(self, text: str, chunk_size: int = 192, overlap: float = 0.85):
        """
        Chunk text with sliding window approach
        Args:
            text: Input text to chunk
            chunk_size: Maximum tokens per chunk
            overlap: Overlap between chunks (0.75 = 75% overlap)
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Calculate stride based on overlap
        stride = int(chunk_size * (1 - overlap))

        chunks = []
        for i in range(0, len(tokens), stride):
            # Get chunk tokens
            chunk_tokens = tokens[i : i + chunk_size]

            # Decode chunk back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            # Stop if we've processed all tokens
            if i + chunk_size >= len(tokens):
                break

        return chunks

    def add_documents(self, documents: List[str]) -> None:
        """
        Process and store documents with embeddings
        Args:
            documents: List of document texts to add
        """
        for doc in documents:
            chunks = self.chunk(doc)
            embeddings = self.base_model.encode(
                chunks,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            for chunk_text, embedding in zip(chunks, embeddings):
                self.chunks.append(
                    {
                        "text": chunk_text,
                        "embedding": embedding,
                        "metadata": {"source_doc": doc[:100]},
                    }
                )

    def search(
        self,
        query: str,
        top_k: int = 5,
        rerank_k: int = 5,
        instruction: Dict[str, str] = INSTRUCTIONS["qa"],
    ) -> List[Tuple[str, float]]:
        """
        Search for relevant chunks using semantic similarity
        Args:
            query: Search query
            top_k: Number of results to return
            rerank_k: Number of results to return after reranking
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        if not self.chunks:
            raise ValueError("No documents added. Please add documents first")
        now = time.time()

        query_embedding = self.query_model.encode(
            instruction["query"] + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        chunk_embeddings = np.array([chunk["embedding"] for chunk in self.chunks])

        similarities = query_embedding @ chunk_embeddings.T

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Build results
        search_results = []
        for i in top_indices:
            search_results.append((self.chunks[i]["text"], float(similarities[i])))

        passages = [result[0] for result in search_results]
        rerank_results = self.rerank(query, passages)
        end = time.time()

        print(f"Time taken: {end - now:.2f}s")

        return rerank_results[:rerank_k]

    def rerank(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        """
        Rerank passages using a cross-encoder
        Args:
            query: Search query
            passages: List of passages to rerank
        Returns:
            List of (passage, relevance_score) tuples
        """
        rankings = self.reranker.rank(query, passages)

        results = []
        for passage, ranking in zip(passages, rankings):
            results.append((passage, float(ranking["score"])))
        return results
