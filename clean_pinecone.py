from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

# --- 配置你的秘钥 ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "marketing-brain"

# --- 初始化 ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print(f"🧹 正在清空索引: {INDEX_NAME} ...")

try:
    # 注意：如果不指定 namespace，它会删除默认 namespace 下的所有数据
    index.delete(delete_all=True)
    print("✅ 旧数据已全部删除！你的索引现在是干净的了。")
except Exception as e:
    print(f"❌ 删除失败: {e}")