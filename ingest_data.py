import os
import time
from pypdf import PdfReader
from pinecone import Pinecone
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Import the BM25 encoder
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "marketing-brain"
DATA_FOLDER = "./data"
BATCH_SIZE = 40

# --- Initialize ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Initialize and fit BM25 Encoder
# For 2GB of data, ideally you'd fit this on your text corpus first.
# For now, we use a default-fit encoder.
bm25 = BM25Encoder.default()

def get_pdf_chunks(file_path):
    """Extracts text from PDF and splits it into chunks."""
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(full_text)

def run_hybrid_ingestion():
    """Ingests data with both Dense and Sparse vectors."""
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Folder {DATA_FOLDER} not found.")
        return

    # IMPORTANT: You must clear the index first because 
    # the schema of your data is changing!
    # --- UPDATED DELETE LOGIC ---
    print("🧹 Attempting to clear index for Hybrid Search upgrade...")
    try:
        # In Serverless, if the index is already empty, this might throw a 404
        index.delete(delete_all=True)
        print("✅ Index cleared.")
    except Exception as e:
        # If it's a 404 (Namespace/Index not found), it's already "clear"
        if "404" in str(e):
            print("💡 Index is already empty or namespace not found. Skipping deletion.")
        else:
            print(f"⚠️ Note during deletion: {e}")
    
    time.sleep(5)

    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.pdf')]
    
    for filename in pdf_files:
        file_path = os.path.join(DATA_FOLDER, filename)
        try:
            chunks = get_pdf_chunks(file_path)
            print(f"\n📄 Processing: {filename} ({len(chunks)} chunks)")

            for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Uploading Hybrid Vectors"):
                batch_text = chunks[i : i + BATCH_SIZE]
                
                # A. Generate Dense Embeddings (Semantic)
                dense_embeddings = pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=batch_text,
                    parameters={"input_type": "passage", "truncate": "END", "dimension": 384}
                )
                
                # B. Generate Sparse Embeddings (Keyword/BM25)
                sparse_embeddings = bm25.encode_documents(batch_text)
                
                vectors = []
                for j, (text, dense_emb, sparse_emb) in enumerate(zip(batch_text, dense_embeddings, sparse_embeddings)):
                    vectors.append({
                        "id": f"{filename}_{i + j}",
                        "values": dense_emb.values,      # Dense vector
                        "sparse_values": sparse_emb,    # Sparse vector (The missing part!)
                        "metadata": {
                            "text": text,
                            "source": filename
                        }
                    })
                
                index.upsert(vectors=vectors)
                time.sleep(1) 

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_hybrid_ingestion()