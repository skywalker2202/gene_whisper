import json
import streamlit as st
import threading
import subprocess
import time
import ollama
import re
import os
# Configuration
FEEDBACK_DIR = "data"
FEEDBACK_PATH = os.path.join(FEEDBACK_DIR, "feedback.json")
DEFAULT_DATASET_PATH = os.path.join("data", "test_kidney_lr_human_human_pmc_line_dict.json")
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'deepseek-r1:1.5b'
TOP_N_RESULTS = 10


# Initialize session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = []
if 'ollama_running' not in st.session_state:
    st.session_state.ollama_running = False
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'current_response' not in st.session_state:
    st.session_state.current_response = {}

def run_ollama_serve():
    """Start Ollama service in the background"""
    if not st.session_state.ollama_running:
        subprocess.Popen(["ollama", "serve"])
        st.session_state.ollama_running = True

def load_dataset():
    """Load and flatten dataset from JSON file as per your format"""
    try:
        with open(DEFAULT_DATASET_PATH) as f:
            data_dict = json.load(f)
        dataset = []
        for v in data_dict.values():
            dataset.extend(v)
        dataset = [i for i in dataset if i]
        dataset = list(set(dataset))
        return dataset
    except FileNotFoundError:
        st.error(f"Error: Dataset file not found at {DEFAULT_DATASET_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

def build_vector_db(dataset):
    """Build vector database with embeddings"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, chunk in enumerate(dataset):
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        st.session_state.vector_db.append((chunk, embedding))
        progress_bar.progress((i+1)/len(dataset))
        status_text.text(f"Processing chunk {i+1}/{len(dataset)}")
    progress_bar.empty()
    status_text.empty()

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

def retrieve(query):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = [
        (chunk, cosine_similarity(query_embedding, embedding))
        for chunk, embedding in st.session_state.vector_db
    ]
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:TOP_N_RESULTS]

def format_prompt(retrieved_chunks):
    context = '\n'.join([f' - {chunk}' for chunk, _ in retrieved_chunks])
    return f'''You are a helpful medical research assistant. Use this context to answer:
{context}

Important:
- Start your answer with confidence percentage in brackets like [85%]
- Never invent facts outside the context
- Keep answers concise and factual
'''

def save_feedback():
    """Save feedback to JSON file as a list of dicts"""
    try:
        with open(FEEDBACK_PATH, 'w') as f:
            json.dump(list(st.session_state.feedback.values()), f, indent=2)
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

def load_feedback():
    """Load existing feedback from JSON file"""
    try:
        with open(FEEDBACK_PATH, 'r') as f:
            feedback_list = json.load(f)
            for idx, fb in enumerate(feedback_list):
                st.session_state.feedback[idx] = fb
    except FileNotFoundError:
        pass  # First run, no feedback file yet

def parse_confidence(answer):
    """Extract confidence percentage from answer"""
    match = re.search(r'\[(\d+)%\]', answer)
    return int(match.group(1)) if match else 100

# Load existing feedback on startup
load_feedback()

# Streamlit UI
st.title("🧬 Kidney Research Chatbot")

# Start Ollama in background thread
threading.Thread(target=run_ollama_serve).start()
time.sleep(2)

# Load and process data
if not st.session_state.vector_db:
    dataset = load_dataset()
    build_vector_db(dataset)
    st.success("Dataset processed!")

# Display chat messages
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            cols = st.columns([4,1,1])
            with cols[1]:
                if st.button("👍", key=f"up_{index}"):
                    if index in st.session_state.current_response:
                        st.session_state.feedback[index] = {
                            **st.session_state.current_response[index],
                            'feedback': 'helpful'
                        }
                        save_feedback()
                        st.toast("Thanks for your feedback! 😺")
            with cols[2]:
                if st.button("👎", key=f"down_{index}"):
                    if index in st.session_state.current_response:
                        st.session_state.feedback[index] = {
                            **st.session_state.current_response[index],
                            'feedback': 'not helpful'
                        }
                        save_feedback()
                        st.toast("We'll improve! 😿")

# Chat input
if prompt := st.chat_input("Ask about kidney research..."):
    if not st.session_state.vector_db:
        st.warning("Dataset not loaded!")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            context = retrieve(prompt)
        prompt_template = format_prompt(context)
        response = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': prompt_template},
                {'role': 'user', 'content': prompt}
            ],
            stream=True,
        )
        response_container = st.empty()
        full_response = ""
        for chunk in response:
            full_response += chunk['message']['content']
            response_container.markdown(full_response + "▌")
        response_container.markdown(full_response)
        # Store response metadata
        response_index = len(st.session_state.messages)
        confidence = parse_confidence(full_response)
        st.session_state.current_response[response_index] = {
            "question": prompt,
            "retrieved_answers": [(chunk, float(score)) for chunk, score in context],
            "generated_answer": full_response,
            "confidence": confidence
        }
        # Feedback buttons
        cols = st.columns([4,1,1])
        with cols[1]:
            if st.button("👍", key=f"up_{response_index}"):
                st.session_state.feedback[response_index] = {
                    **st.session_state.current_response[response_index],
                    'feedback': 'helpful'
                }
                save_feedback()
                st.toast("Thanks for your feedback! 😺")
        with cols[2]:
            if st.button("👎", key=f"down_{response_index}"):
                st.session_state.feedback[response_index] = {
                    **st.session_state.current_response[response_index],
                    'feedback': 'not helpful'
                }
                save_feedback()
                st.toast("We'll improve! 😿")
    st.session_state.messages.append({"role": "assistant", "content": full_response})
