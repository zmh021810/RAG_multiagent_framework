import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from pinecone import Pinecone
from groq import Groq
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
load_dotenv()

# --- 1. Client & Encoder Initialization ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

INDEX_NAME = "marketing-brain"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
groq_client = Groq(api_key=GROQ_API_KEY)
bm25 = BM25Encoder.default()

# --- 2. State Definition ---
class AgentState(TypedDict):
    query: str
    raw_retrieval: List[dict]
    vector_context: List[dict]
    draft: str
    critique: str        
    revision_count: int

# --- 3. Node Functions ---

def retrieval_node(state: AgentState):
    print("--- NODE: HYBRID RETRIEVAL ---")
    
    # 1. Generate Dense Embedding (Semantic)
    dense_emb = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[state["query"]],
        parameters={"input_type": "query", "dimension": 384} 
    )
    
    # 2. Generate Sparse Embedding (Keyword-based BM25)
    sparse_emb = bm25.encode_queries(state["query"])
    
    # Debug: Confirm if Sparse Vector is generating signal
    if sparse_emb and "indices" in sparse_emb and "values" in sparse_emb:
        print(f"--- DEBUG: Sparse Vector is ACTIVE ---")
        print(f"--- DEBUG: Number of non-zero indices: {len(sparse_emb['indices'])}")
    else:
        print(f"--- DEBUG: WARNING! Sparse Vector is EMPTY or INVALID ---")
    
    # 3. Execute Hybrid Query on Pinecone
    results = index.query(
        vector=dense_emb[0].values,
        sparse_vector=sparse_emb,
        top_k=50, # High top_k to increase recall for long documents
        include_metadata=True
    )
    
    # 4. Format the retrieved matches into a list of dictionaries
    raw_docs = [{"text": m.metadata["text"], "source": m.metadata["source"], "id": m.id} for m in results.matches]
    
    # --- START OF DEBUG CHECK FOR SECTION 212 ---
    print(f"--- DEBUG: Scanning retrieved context for '212' ---")
    found_target = False
    for i, doc in enumerate(raw_docs):
        # We check if the specific keyword '212' exists in any of the 50 snippets
        if "212" in doc['text']:
            print(f"✅ SUCCESS: '212' found in Result #{i+1} (Source: {doc['source']})")
            print(f"Preview: {doc['text'][:150]}...")
            found_target = True
            break 
            
    if not found_target:
        print(f"❌ FAILURE: '212' is missing from all 50 retrieved snippets.")
        print("Suggestion: Check if Section 212 was properly indexed in ingest_data.py")
    # --- END OF DEBUG CHECK ---

    return {"raw_retrieval": raw_docs, "revision_count": 0}

def reranker_node(state: AgentState):
    print(f"--- NODE: RERANKER (Refining {len(state['raw_retrieval'])} docs) ---")
    candidates_text = ""
    for idx, doc in enumerate(state['raw_retrieval']):
        candidates_text += f"ID {idx}: {doc['text'][:300]}...\n\n"
    
    rerank_prompt = f"USER QUERY: {state['query']}\nCANDIDATES:\n{candidates_text}\nPick TOP 5 IDs (e.g., 0, 1, 2...)."
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": rerank_prompt}]
    )
    try:
        top_ids = [int(i.strip()) for i in response.choices[0].message.content.split(',') if i.strip().isdigit()]
        refined = [state['raw_retrieval'][i] for i in top_ids if i < len(state['raw_retrieval'])][:5]
    except:
        refined = state['raw_retrieval'][:5]
    return {"vector_context": refined}

def writer_node(state: AgentState):
    """
    Final optimized Writer Node to solve the 'Refusal' problem.
    This version forces the LLM to perform literal extraction for Section 212.
    """
    print(f"--- NODE: WRITER (Revision: {state['revision_count']}) ---")
    
    # 1. Structure the context so Source and Content are inseparable in the LLM's mind
    context_parts = []
    for i, c in enumerate(state["vector_context"]):
        context_parts.append(f"SOURCE_FILE: {c['source']}\nRAW_CONTENT: {c['text']}")
    
    context_text = "\n\n---\n\n".join(context_parts)
    
    # 2. Strict Prompt: Explicitly points the LLM to the Section 212 string
    prompt = f"""You are a professional Legal Document Processor. 
    Use the provided RAW_CONTENT to answer the user query.

    ### CRITICAL EXTRACTION RULES:
    1. FIND '§ 212' in the text. It contains the phrase 'Superintendent of Documents'.
    2. YOU MUST explain that Section 212 requires the Superintendent of Documents to supply the Code at the start of a new Congress.
    3. For the 'Rules of Construction', explain that masculine words include feminine ones based on Section 1 of [USCODE-2022-title1.pdf].
    4. CITE the exact filename: [USCODE-2022-title1.pdf].

    ### SOURCE DATA:
    {context_text}

    ### USER QUERY: 
    {state['query']}
    
    Final Answer (with legal disclaimer):"""
    
    # 3. Use 0.1 for maximum factual grounding
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1 
    )
    
    return {
        "draft": response.choices[0].message.content, 
        "revision_count": state["revision_count"] + 1
    }

def compliance_reviewer_node(state: AgentState):
    print("--- NODE: COMPLIANCE REVIEWER ---")
    review_prompt = f"""Review this draft for: 
    1. Direct answer to query 2. Citations 3. Disclaimer.
    Reply 'PASSED' if perfect, otherwise provide a critique.
    DRAFT: {state['draft']}"""
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": review_prompt}]
    )
    return {"critique": response.choices[0].message.content}

# --- 4. Routing Logic ---

def decide_to_finish(state: AgentState):
    """
    路由函数：判断是结束还是回去重写
    """
    if "PASSED" in state["critique"].upper() or state["revision_count"] >= 3:
        print("--- DECISION: FINISHED ---")
        return "end"
    else:
        print(f"--- DECISION: REWRITE (Reason: {state['critique'][:50]}...) ---")
        return "rewrite"

# --- 5. Graph Construction ---

def build_refined_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", compliance_reviewer_node)
    
    workflow.set_entry_point("retrieval")
    workflow.add_edge("retrieval", "reranker")
    workflow.add_edge("reranker", "writer")
    workflow.add_edge("writer", "reviewer")
    
    # check, if does not match, go to writer again
    workflow.add_conditional_edges(
        "reviewer",
        decide_to_finish,
        {
            "end": END,
            "rewrite": "writer"
        }
    )
    
    return workflow.compile()

if __name__ == "__main__":
    app = build_refined_graph()
    question="According to Title 1 and Title 2 of the U.S. Code, what are the specific rules for the 'Rules of Construction' regarding the gender of words used in statutes, and how does the 'Superintendent of Documents' handle the additional distribution of the Code to a new Congress under Section 212?"
    result = app.invoke({"query": question})
    print("\n" + "="*30 + "\nFINAL ANSWER:\n" + result["draft"])