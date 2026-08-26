import os

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore


docs = load_all_documents("data")


# -------------------------
# Semantic FAISS
# -------------------------

semantic_path = "faiss_store/faiss.index"


semantic_store = FaissVectorStore(
    persist_dir="faiss_store",
    semantic=True
)


if os.path.exists(semantic_path):

    print(
        "[INFO] Loading semantic FAISS"
    )

    semantic_store.load()

else:

    print(
        "[INFO] Building semantic FAISS"
    )

    semantic_store.build_from_documents(
        docs
    )



# -------------------------
# Recursive FAISS
# -------------------------

recursive_path = "faiss_store_recursive/faiss.index"


recursive_store = FaissVectorStore(
    persist_dir="faiss_store_recursive",
    semantic=False
)


if os.path.exists(recursive_path):

    print(
        "[INFO] Loading recursive FAISS"
    )

    recursive_store.load()

else:

    print(
        "[INFO] Building recursive FAISS"
    )

    recursive_store.build_from_documents(
        docs
    )


print(
    "[INFO] Both FAISS stores ready"
)