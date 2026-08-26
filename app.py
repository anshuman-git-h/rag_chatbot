from src.search import RAGSearch


if __name__ == "__main__":

    rag_search = RAGSearch(
        persist_dir="faiss_store",
        use_hyde=True
    )


    while True:

        query = input(
            "\nEnter query: "
        ).strip()


        if query.lower()=="exit":
            break


        result = rag_search.search_and_summarize(
            query,
            top_k=3
        )


        print(
            "\nAnswer:",
            result["answer"]
        )