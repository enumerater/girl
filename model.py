from langchain_community.chat_models import ChatTongyi

import dotenv
import os

from langchain_community.embeddings import DashScopeEmbeddings

dotenv.load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")

model = ChatTongyi(api_key=api_key)

# 向量模型
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=api_key
)

