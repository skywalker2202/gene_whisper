

import threading
import subprocess
import time
import ollama

# Configuration
DATASET_PATH = '/Users/suchanda/Desktop/workspace_rwth/rag/notebooks/cat/cat-facts.txt'
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'deepseek-r1:1.5b'
TOP_N_RESULTS = 3

# Global vector database (chunk, embedding tuples)
VECTOR_DB = []

def run_ollama_serve():
    """Start Ollama service in the background"""
    subprocess.Popen(["ollama", "serve"])

def load_dataset():
    """Load cat facts dataset from file"""
    try:
        with open(DATASET_PATH, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        exit(1)

def build_vector_db(dataset):
    """Build vector database with embeddings"""
    print(f"\n🔍 Building vector database with {len(dataset)} entries...")
    for i, chunk in enumerate(dataset):
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        VECTOR_DB.append((chunk, embedding))
        print(f"  ✅ Processed chunk {i+1}/{len(dataset)}")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

def retrieve(query):
    """Retrieve relevant context using vector similarity"""
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    
    similarities = [
        (chunk, cosine_similarity(query_embedding, embedding))
        for chunk, embedding in VECTOR_DB
    ]
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:TOP_N_RESULTS]

def format_prompt(retrieved_chunks):
    """Create system prompt with context"""
    context = '\n'.join([f' - {chunk}' for chunk, _ in retrieved_chunks])
    return f'''You are a helpful cat fact expert. Use this context to answer:
{context}

Important:
- Mention confidence level based on context similarity
- Never invent facts outside the context
- Keep answers concise and factual
'''

def main():
    """Main chatbot interaction loop"""
    # Start Ollama service
    print("🚀 Starting Ollama service...")
    threading.Thread(target=run_ollama_serve).start()
    time.sleep(5)  # Allow service initialization

    # Load and process data
    dataset = load_dataset()
    build_vector_db(dataset)

    # Chat interface
    print("\n🐱 Welcome to Cat Fact Chatbot! Ask me anything about cats!")
    while True:
        try:
            query = input("\n🤔 Your question: ").strip()
            if not query:
                continue
                
            # Retrieve relevant context
            context = retrieve(query)
            
            # Generate response
            response = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[
                    {'role': 'system', 'content': format_prompt(context)},
                    {'role': 'user', 'content': query}
                ],
                stream=True,
            )
            
            # Stream response
            print("\n💡 Answer:")
            for chunk in response:
                print(chunk['message']['content'], end='', flush=True)
            print()  # New line after response

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
