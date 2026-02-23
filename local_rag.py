import os
import tempfile
import streamlit as st

# Local RAG & LLM Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# ---------------------------------------------------------
# SYSTEM INITIALIZATION: 100% LOCAL AI
# ---------------------------------------------------------
# Connect to the local Ollama instance running on your machine
# You can change this to "qwen2.5" if you pulled that model instead
LOCAL_MODEL_NAME = "llama3.1"

try:
    print(f"System Log: Connecting to local Ollama model ({LOCAL_MODEL_NAME})...")
    llm = Ollama(model=LOCAL_MODEL_NAME, temperature=0.2)
except Exception as e:
    st.error("Fatal Error: Could not connect to Ollama. Ensure Ollama is installed and running.")
    st.stop()

# UI Setup
st.set_page_config(page_title="Offline Legal AI", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "last_petition" not in st.session_state:
    st.session_state.last_petition = None

# ---------------------------------------------------------
# CORE ENGINE: LOCAL KNOWLEDGE BASE
# ---------------------------------------------------------
def build_local_knowledge_base(uploaded_files):
    """
    Ingests legal PDFs and stores them in local ChromaDB.
    Zero external API calls are made during this process.
    """
    print("System Log: Compiling Local Vector Database...")
    documents = []
    
    try:
        for uploaded_file in uploaded_files:
            file_ext = os.path.splitext(uploaded_file.name)[1]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name
                
            print(f"System Log: Parsing document -> {uploaded_file.name}")
            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())
            os.remove(temp_path)

        if not documents:
            return None

        print("System Log: Chunking text for local embedding...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        print("System Log: Generating embeddings locally via HuggingFace...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        vector_db = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="./chroma_local_db")
        
        print("System Log: Offline Database Compilation Successful!")
        return vector_db
        
    except Exception as e:
        print(f"System Log: Database failure -> {e}")
        return None

def stream_local_generator(prompt_text):
    """Streams the response chunk-by-chunk from local Ollama."""
    for chunk in llm.stream(prompt_text):
        yield chunk

# ---------------------------------------------------------
# FRONTEND: SIDEBAR CONFIGURATION
# ---------------------------------------------------------
with st.sidebar:
    st.title("🔒 Privacy-First RAG")
    st.success("Status: 100% Offline & Free")
    
    rulebook_files = st.file_uploader("Upload Legal PDFs", type=['pdf'], accept_multiple_files=True)
    
    if st.button("Compile Local Database"):
        if rulebook_files:
            with st.spinner("Processing documents locally. No internet required..."):
                db = build_local_knowledge_base(rulebook_files)
                if db:
                    st.session_state.vector_db = db
                    st.success("Local Database Locked & Loaded!")
                else:
                    st.error("Compilation failed.")
        else:
            st.warning("Please upload PDFs first.")

    if st.button("Purge Session Memory"):
        st.session_state.messages = []
        st.session_state.last_petition = None
        st.rerun()

# ---------------------------------------------------------
# FRONTEND: MAIN CHAT INTERFACE
# ---------------------------------------------------------
st.title("⚖️ Autonomous Legal Counsel")

if not st.session_state.vector_db:
    st.warning("⚠️ Knowledge Base Offline. Upload documents to begin.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your legal issue here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.vector_db:
            st.error("System requires a compiled Knowledge Base.")
        else:
            print(f"System Log: Performing semantic search for -> '{prompt[:30]}...'")
            relevant_chunks = st.session_state.vector_db.similarity_search(prompt, k=4)
            retrieved_context = "\n\n".join([f"DOCUMENT EXCERPT:\n{c.page_content}" for c in relevant_chunks])
            
            # Prune payload to optimize local processing power
            clean_history = ""
            for msg in st.session_state.messages[-4:]:
                safe_content = msg["content"].split("LOCAL KNOWLEDGE BASE:")[0]
                clean_history += f"{msg['role'].upper()}: {safe_content}\n"

            final_payload = f"""Sen Türkiye'de görev yapan uzman bir üniversite hukuku asistanısın. 
Aşağıdaki 'LOCAL KNOWLEDGE BASE' metinlerini kullanarak öğrencinin sorununa çözüm bul. Maddeleri referans göster. Türkçe cevap ver.

Chat History:
{clean_history}

LOCAL KNOWLEDGE BASE:
{retrieved_context}

User Query: {prompt}
Answer:"""
            
            try:
                print("System Log: Dispatching prompt to local Ollama Engine...")
                # Stream the response natively
                full_response = st.write_stream(stream_local_generator(final_payload))
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Local Inference Error: {e}")

# ---------------------------------------------------------
# FRONTEND: BACKGROUND TASKS
# ---------------------------------------------------------
if len(st.session_state.messages) > 1:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    clean_history = "\n".join([f"{msg['role'].upper()}: {msg['content'].split('LOCAL KNOWLEDGE BASE:')[0]}" for msg in st.session_state.messages])

    with col1:
        if st.button("📝 Draft Legal Petition"):
            with st.spinner("Local AI is compiling your petition..."):
                petition_prompt = f"Aşağıdaki konuşma geçmişine dayanarak, üniversite dekanlığına hitaben resmi ve hukuki bir dilekçe (petition) yaz. Sadece dilekçe metnini ver.\n\nCLEAN CHAT HISTORY:\n{clean_history}"
                try:
                    res = llm.invoke(petition_prompt)
                    st.session_state.last_petition = res
                    st.subheader("Petition Draft")
                    st.markdown(res)
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if st.button("⚖️ Anti-Thesis Analysis"):
            if st.session_state.last_petition:
                with st.spinner("Adversarial AI analyzing vulnerabilities..."):
                    anti_prompt = f"Sen üniversitenin karşı avukatısın. Aşağıdaki dilekçedeki hukuki açıkları, eksik usulleri ve reddedilme gerekçelerini sert bir dille yaz.\n\nPetition:\n{st.session_state.last_petition}"
                    try:
                        anti_res = llm.invoke(anti_prompt)
                        st.subheader("⚠️ Risk Report")
                        st.warning(anti_res)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Generate a petition first.")