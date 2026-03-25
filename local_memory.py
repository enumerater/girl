from langchain_chroma import Chroma

from model import embedding_model

# 真实回忆存在这里，不和聊天记忆混在一起
LOCAL_MEMORY_DIR = "./local_memory_db"

# 加载真实回忆（只会初始化一次，永远不变）
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory=LOCAL_MEMORY_DIR,
)