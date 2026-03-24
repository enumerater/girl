from langchain_community.chat_models import ChatTongyi
import dotenv
import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma  # 注意：这里变了！

dotenv.load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")

model = ChatTongyi(api_key=api_key, model_name="qwen-plus")

# 向量模型
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=api_key
)

# 本地文件夹，所有记忆都会存在这里，永久不丢
PERSIST_DIRECTORY = "./chroma_memory"

# 加载或创建向量库
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory=PERSIST_DIRECTORY,  # 关键：持久化保存
)
