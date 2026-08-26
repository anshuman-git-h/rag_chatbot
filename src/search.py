import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant",use_hyde = True):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.vectorstore.load()
        else:
            raise Exception(
                f"FAISS store missing: {persist_dir}. Run build_vectorstores.py first."
            )
        
        groq_api_key = "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"
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        self.use_hyde = use_hyde
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def generate_hyde_document(self, query: str) -> str:
        prompt = f"""
        Write a short, 2-4 sentence paragraph in the plain, direct style of a school textbook that would answer this question. Do not include headers, section titles, bullet points, abstracts, or keyword lists. Do not cite fictional papers or use academic formatting

        Question:
        {query} 
        Hypothetical Document:
        """

        response = self.llm.invoke([prompt])
        return response.content
    
    def search_and_summarize(self, query: str, top_k: int = 5) -> dict:


        if not self.use_hyde:
            search_query = query
        else:
            search_query = self.generate_hyde_document(query)
            print("[INFO] HyDE generated:")
            print(search_query[:300] + "...")

        results = self.vectorstore.query(search_query, top_k=top_k)

        texts = [r["metadata"].get("text", "")
                  for r in results 
                  if r["metadata"]
                  ]
        context = "\n\n".join(texts)

        if not context:
            return {
                "answer": "No relevant documents found.",
                "context": []
            }
        
        prompt = f"""Answer the query using the provided context.

                    Query:
                    {query}

                    Context:
                    {context}

                    Answer:
                    """

        response = self.llm.invoke([prompt])

        return {
            "answer": response.content,
            "context": texts
        }

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "The prevalence of hidden suppression"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)