import os
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma

# 配置参数（与你构建时保持一致）
PERSIST_DIRECTORY = "./chroma_db"
CHROMA_COLLECTION = 'vue3_components_docs'  # 你的collection名称

# Step 1. 载入embedding模型和embedding接口
embedder = SentenceTransformer('all-MiniLM-L6-v2')
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()
embedding_function = SentenceTransformerEmbeddings(embedder)

# Step 2. 载入Chroma向量数据库
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_function,
    collection_name=CHROMA_COLLECTION
)

# Step 3. 定义你要检索的查询文本
query = input("请输入检索内容：\n")  # 例："如何自定义Button组件的样式？"

# Step 4. 检索最相似的top_k个文档
top_k = 200
results = vectorstore.similarity_search_with_score(query, k=top_k)

# Step 5. 打印结果
for i, (doc, score) in enumerate(results):
    print(f"\n--- Top {i+1} ---")
    print(f"得分（越低越相关）: {score:.4f}")
    print(f"元数据: {doc.metadata}")
    print(f"正文内容: {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''}")

print(f"\n共返回 {len(results)} 条结果。")
