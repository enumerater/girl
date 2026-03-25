from langchain_chroma import Chroma
import os
from model import embedding_model

REAL_MEMORY_DIR = "./real_memory_db"

real_vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory=REAL_MEMORY_DIR,
)