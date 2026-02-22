import os
import tempfile
import streamlit as st
from google import genai
from google.genai import types

# RAG (Vector DB) Core Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. API CONFIGURATION & SECURITY
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("Fatal Error: GEMINI_API_KEY environment variable is missing.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# Using the standard flash model to avoid the caching quota limits
MODEL_ID = 'gemini-1.5-flash-002'

# 2. UI INITIALIZATION & SESSION MEMORY
st.set_page_config(page_title="LegalTech Agentic RAG", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "last_petition" not in st.session_state:
    st.session_state.last_petition = None

# ---------------------------------------------------------
# CORE ENGINE: AGENTIC HYBRID KNOWLEDGE BASE
# ---------------------------------------------------------
def build_hybrid_knowledge_base(uploaded_files):
    """
    Ingests multiple PDF documents (e.g., YÖK and specific university regulations).
    Chunks and embeds them into a local Chroma Vector Database.
    This acts as the 'Static Knowledge' for the AI Agent.
    """
    print("System Log: Initializing Hybrid Knowledge Base...")
    documents = []
    
    try:
        # Iterate through all uploaded files and parse them
        for uploaded_file in uploaded_files:
            file_ext = os.path.splitext(uploaded_file.name)[1]
            
            # Secure temporary storage for parsing
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name
                
            print(f"System Log: Loading document into memory -> {uploaded_file.name}")
            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())
            
            # Clean up local storage
            os.remove(temp_path)

        if not documents:
            print("System Log: No readable text found in documents.")
            return None

        print("System Log: Executing Recursive Character Chunking algorithm...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        print("System Log: Downloading/Loading HuggingFace Embedding Model...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        print("System Log: Compiling Vector Database (ChromaDB)...")
        vector_db = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="./chroma_agent_db")
        
        print("System Log: Knowledge Base compilation successful!")
        return vector_db
        
    except Exception as e:
        print(f"System Log: Critical failure in Vector DB compilation: {e}")
        return None

def stream_generator(response_stream):
    """Utility generator to stream response chunks to the Streamlit frontend."""
    for chunk in response_stream:
        if chunk.text:
            yield chunk.text

# ---------------------------------------------------------
# FRONTEND: SIDEBAR CONFIGURATION
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Agentic Router Config")
    st.info("Upload static regulations (YÖK, University rules). The AI will automatically route dynamic queries (Mobbing, TCK) to Google Search.")
    
    # Allow multiple files for combined YÖK + School rulebook injection
    rulebook_files = st.file_uploader("Upload Regulations (PDF)", type=['pdf'], accept_multiple_files=True)
    
    if st.button("Compile Knowledge Base"):
        if rulebook_files:
            with st.spinner("Embedding documents into Vector Space..."):
                db = build_hybrid_knowledge_base(rulebook_files)
                if db:
                    st.session_state.vector_db = db
                    st.success("Vector Database Active & Linked!")
                else:
                    st.error("Database compilation failed.")
        else:
            st.warning("Please upload at least one PDF document.")

    if st.button("Purge System Memory"):
        st.session_state.messages = []
        st.session_state.last_petition = None
        st.rerun()

# ---------------------------------------------------------
# FRONTEND: MAIN CHAT & AGENT LOGIC
# ---------------------------------------------------------
st.title("⚖️ Agentic Legal Counsel")

# Render previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user query
if prompt := st.chat_input("Enter your legal issue or scenario here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        # 1. THE ROUTER PROMPT (Crucial for Agentic Behavior)
        sys_instruction = """
        You are an elite legal AI agent operating in Turkey, specializing in higher education law and student rights.
        
        AGENTIC ROUTING INSTRUCTIONS:
        1. LOCAL DATA PRIORITY: If the user's query is about academic procedures (grades, graduation, curriculum, attendance), rely PRIMARILY on the 'LOCAL KNOWLEDGE BASE' text provided below.
        2. LIVE SEARCH TRIGGER: If the query involves crimes, mobbing, constitutional rights, abuse of authority (Görevi Kötüye Kullanma), or requires 'Emsal Kararlar' (legal precedents), YOU MUST utilize the Google Search tool to scan the Turkish Penal Code (TCK) and recent Supreme Court (Yargıtay) rulings.
        3. QUESTIONING PROTOCOL: If critical variables are missing from the user's scenario to make a definitive legal judgment, ASK clarifying questions before providing a final verdict.
        4. Cite specific article numbers in your analysis. Respond entirely in Turkish.
        """
        
        # 2. CONTEXT ASSEMBLY
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        final_payload = f"Chat History:\n{chat_history}\n\n"
        
        # Inject Vector DB chunks if available
        if st.session_state.vector_db:
            print(f"System Log: Querying Vector DB for prompt -> '{prompt[:30]}...'")
            relevant_chunks = st.session_state.vector_db.similarity_search(prompt, k=4)
            retrieved_context = "\n\n".join([f"DOCUMENT EXCERPT:\n{c.page_content}" for c in relevant_chunks])
            final_payload += f"LOCAL KNOWLEDGE BASE (Static Rules):\n{retrieved_context}\n\n"
            
        final_payload += f"User's Current Query: {prompt}"
        
        # 3. LLM EXECUTION
        try:
            print("System Log: Executing Agentic LLM Call (Search + Vector Context)...")
            response_stream = client.models.generate_content_stream(
                model=MODEL_ID,
                contents=final_payload,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    temperature=0.3, # Low temperature minimizes hallucination
                    tools=[types.Tool(google_search=types.GoogleSearch())] # Enables dynamic tool calling
                )
            )
            
            full_response = st.write_stream(stream_generator(response_stream))
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Critical API Error during generation: {e}")
            print(f"System Log: Exception Details -> {e}")

# ---------------------------------------------------------
# FRONTEND: ADVERSARIAL & ACTION BUTTONS
# ---------------------------------------------------------
# Buttons appear only after initial interaction
if len(st.session_state.messages) > 1:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Generate Official Petition"):
            with st.spinner("Drafting formal legal petition..."):
                petition_prompt = "Based on the entire chat history, draft a formal, legally structured petition (dilekçe) in Turkish addressing the university dean's office or relevant authority."
                
                print("System Log: Executing background task -> Petition Generation")
                res = client.models.generate_content(
                    model=MODEL_ID,
                    contents=petition_prompt + "\n" + str(st.session_state.messages),
                    config=types.GenerateContentConfig(temperature=0.2)
                )
                st.session_state.last_petition = res.text
                st.subheader("Official Petition Draft")
                st.markdown(res.text)

    with col2:
        if st.button("⚖️ Execute Anti-Thesis Analysis"):
            if st.session_state.last_petition:
                with st.spinner("Adversarial AI analyzing petition for loopholes..."):
                    anti_prompt = f"Act as the university's strict legal counsel. Critique the following petition. Find legal loopholes, missing procedural steps, and state exact reasons why it could be rejected. \nPetition:\n{st.session_state.last_petition}"
                    
                    print("System Log: Executing background task -> Adversarial Anti-Thesis")
                    anti_res = client.models.generate_content(
                        model=MODEL_ID,
                        contents=anti_prompt,
                        config=types.GenerateContentConfig(temperature=0.6) # Higher temp for adversarial creativity
                    )
                    st.subheader("⚠️ Anti-Thesis Risk Report")
                    st.warning(anti_res.text)
            else:
                st.error("Action Denied: Generate a petition first to analyze it.")
                
    with col3:
        if st.button("📨 Generate State Complaint (CİMER/YÖK)"):
            with st.spinner("Drafting high-level complaint to state authorities..."):
                comp_prompt = "Based on the chat history, draft a formal complaint in Turkish directed to YÖK or CİMER. Emphasize concepts like 'Görevi Kötüye Kullanma' (Abuse of Authority) and systematic rights violations if applicable."
                
                print("System Log: Executing background task -> State Complaint Generation")
                comp_res = client.models.generate_content(
                    model=MODEL_ID,
                    contents=comp_prompt + "\n" + str(st.session_state.messages),
                    config=types.GenerateContentConfig(temperature=0.2)
                )
                st.subheader("State Authority Complaint")
                st.markdown(comp_res.text)