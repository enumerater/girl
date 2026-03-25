from langchain_chroma import Chroma
import os
from model import embedding_model

# 真实回忆存在这里
REAL_MEMORY_DIR = "./real_memory_db"


def import_real_memories_from_txt():
    if not os.path.exists(REAL_MEMORY_DIR):
        print("🔍 首次运行，开始导入真实回忆...")

        real_vector_store = Chroma(
            embedding_function=embedding_model,
            persist_directory=REAL_MEMORY_DIR,
        )

        # 读取你的回忆文本
        with open("real_memories.txt", "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            real_vector_store.add_texts(lines)

        print("✅ 真实回忆已永久导入，永不改变")
    else:
        print("✅ 真实回忆库已存在，跳过导入")

if __name__ == "__main__":
    import_real_memories_from_txt()