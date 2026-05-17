import chromadb
import ollama

# ─────────────────────────────
# STEP 1: Load Medical Knowledge
# ─────────────────────────────
print("🔄 Step 1: Loading medical knowledge...")
with open("medical_knowledge.txt", "r") as f:
    text = f.read()
print(f"✅ Loaded: {len(text)} characters")

# ─────────────────────────────
# STEP 2: Store in Vector DB
# ─────────────────────────────
print("🔄 Step 2: Setting up vector DB...")
client = chromadb.Client()
collection = client.create_collection("medical_docs")

chunks = [text[i:i+500] for i in range(0, len(text), 500)]
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print(f"✅ Stored {len(chunks)} chunks in vector DB")

# ─────────────────────────────
# STEP 3: Q&A Loop
# ─────────────────────────────
print("\n💬 Medical RAG Q&A Ready!")
print("📋 Example: 'patient has chest pain and shortness of breath'")
print("Type 'exit' to quit\n")

while True:
    question = input("🩺 Enter symptoms: ")
    
    if question.lower() == "exit":
        print("👋 Goodbye!")
        break
    
    # Find relevant medical context
    print("🔍 Searching medical knowledge base...")
    results = collection.query(
        query_texts=[question],
        n_results=2
    )
    context = " ".join(results['documents'][0])
    
    # Ask Ollama
    print("🤖 Analyzing with Ollama...")
    response = ollama.chat(
        model='qwen3.5:4b',
        messages=[{
            'role': 'user',
            'content': f"""You are a medical assistant. Based only on this medical knowledge:

{context}

A patient has these symptoms: {question}

Provide:
1. Possible condition
2. Severity level (Low/Medium/High/Critical)
3. Recommended action

Be clear and concise."""
        }]
    )
    
    print(f"\n📋 Assessment:\n{response['message']['content']}")
    print("\n" + "─"*50 + "\n")