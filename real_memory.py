from langchain_chroma import Chroma

from model import embedding_model

# 真实回忆存在这里，不和聊天记忆混在一起
REAL_MEMORY_DIR = "./real_memory_db"

# 加载真实回忆（只会初始化一次，永远不变）
real_vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory=REAL_MEMORY_DIR,
)


# 只有第一次运行时执行：把 txt 导入
def import_real_memories_from_txt():
    import os
    if not os.path.exists(REAL_MEMORY_DIR):
        with open("real_memories.txt", "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # 一次性导入，永不修改、永不添加
        real_vector_store.add_texts(lines)
        print("✅ 真实回忆已永久导入，永不改变")


# 执行导入（只执行一次！以后永远不要动！）
import_real_memories_from_txt()