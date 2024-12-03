from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np


class RAG:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        reranker_name: str = "BAAI/bge-ranker-base",
    ):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Dict] = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reranker = CrossEncoder(reranker_name)

        self.chunks = []
        self.bm25 = None

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

    def add_documents(
        self, documents: List[str], metadata: Optional[List[Dict]]
    ) -> None:
        """
        Process and store documents with embeddings
        Args:
            documents: List of document texts to add
        """

        if metadata is None:
            metadata = [{"source": f"doc_{i}"} for i in range(len(documents))]

        all_chunk_texts = []

        for doc, meta in zip(documents, metadata):
            chunks = self.chunk(doc)

            embeddings = self.model.encode(
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

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for relevant chunks using semantic similarity
        Args:
            query: Search query
            top_k: Number of results to return
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        if not self.chunks:
            raise ValueError("No documents added. Please add documents first")

        instruction = "Represent this sentence for searching relevant passages: "
        query_embedding = self.model.encode(
            instruction + query, normalize_embeddings=True, convert_to_numpy=True
        )

        chunk_embeddings = np.array([chunk["embedding"] for chunk in self.chunks])
        similarities = np.dot(chunk_embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Build results
        results = []
        for i in top_indices:
            results.append((self.chunks[i]["text"], float(similarities[i])))

        return results
