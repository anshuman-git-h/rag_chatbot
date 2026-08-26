from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents
from sklearn.metrics.pairwise import cosine_similarity
import nltk
try:
  nltk.data.find("tokenizers/punkt_tab")
except LookupError:
  nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize
from langchain_core.documents import Document


class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200, similarity_threshold: float = 0.6):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        chunks = []
        for doc in documents:
            # Split into sentences
            sentences = sent_tokenize(doc.page_content)
            if not sentences:
                continue
            # Generate embeddings for each sentence
            embeddings = self.model.encode(sentences)
            current_chunk = [sentences[0]]
            for i in range(1, len(sentences)):
                similarity = cosine_similarity([embeddings[i - 1]],[embeddings[i]])[0][0]

                # Split if semantic similarity drops
                if similarity < self.similarity_threshold:
                    chunks.append(
                        Document(
                            page_content=" ".join(current_chunk),
                            metadata=doc.metadata
                        )
                    )
                    current_chunk = [sentences[i]]
                else:
                    current_chunk.append(sentences[i])

            # Add the last chunk
            if current_chunk:
                chunks.append(
                    Document(
                        page_content=" ".join(current_chunk),
                        metadata=doc.metadata
                    )
                )

        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} semantic chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)