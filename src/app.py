import streamlit as st
import os
import base64
from dotenv import load_dotenv

load_dotenv()

from ingest import get_or_create_chunks
from retriever import HybridRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Safety & Compliance Copilot", page_icon="🛡️", layout="wide")

# Helper to load and encode local background image as base64
def get_image_base64(path):
    for p in [path, path.lstrip("/"), os.path.join(os.getcwd(), path.lstrip("/"))]:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    return ""

img_b64 = get_image_base64("images/minesafe_wp.png")

bg_style = f"""
    .stApp {{
        background-color: #050811;
        background-image: linear-gradient(rgba(5, 8, 17, 0.45), rgba(5, 8, 17, 0.65)), url("data:image/png;base64,{img_b64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stSidebar"] {{
        background-color: #0b1120 !important;
        border-right: 1px solid #1e293b;
    }}
    /* Remove default white top header bar in Streamlit */
    [data-testid="stHeader"] {{
        background: transparent !important;
    }}
""" if img_b64 else ".stApp { background-color: #050811; }"

st.markdown(f"""
    <style>
    {bg_style}
    /* Global Crisp Text Contrast */
    h1, h2, h3, h4, h5, h6, p, span, label, div {{
        color: #f8fafc !important;
    }}
    .main-header {{
        background: rgba(15, 23, 42, 0.90);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(234, 179, 8, 0.4);
        padding: 2.2rem 2.5rem;
        border-radius: 14px;
        margin-top: -1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.5);
    }}
    .header-content {{
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }}
    .header-icon {{
        font-size: 3rem;
        background: rgba(250, 204, 21, 0.15);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(250, 204, 21, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .main-header h1 {{
        color: #f8fafc !important;
        font-size: 2.25rem;
        margin-bottom: 0.4rem;
        letter-spacing: -0.025em;
    }}
    .main-header p {{
        color: #facc15 !important;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 0;
    }}
    .telemetry-card {{
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    .telemetry-label {{
        font-size: 0.75rem;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .telemetry-value {{
        font-size: 0.95rem;
        font-weight: 600;
        color: #facc15 !important;
    }}
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 16px;
        background-color: rgba(30, 41, 59, 0.95);
        backdrop-filter: blur(10px);
        padding: 10px 16px;
        border-radius: 8px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 45px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #cbd5e1 !important;
        font-weight: 600;
        font-size: 1rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: rgba(250, 204, 21, 0.15) !important;
        color: #facc15 !important;
        border-bottom: 2px solid #facc15;
    }}
    .stButton button {{
        border-radius: 8px;
        font-weight: 500;
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
    }}
    .stButton button:hover {{
        background-color: #334155;
        border-color: #facc15;
        color: #facc15;
    }}
    </style>
    <div class="main-header">
        <div class="header-content">
            <div class="header-icon">🛡️</div>
            <div>
                <h1>Enterprise Safety & Compliance Copilot</h1>
                <p>⚡ AI-Powered Regulatory Search, Hazard Guidance, and Zero-Hallucination Citation Engine</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    chunks = get_or_create_chunks()
    retriever = HybridRetriever(chunks)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    return retriever, llm

retriever, llm = initialize_system()

# Sidebar Telemetry & Status
st.sidebar.success("🟢 System Status: Operational")
st.sidebar.subheader("📊 System Telemetry")

st.sidebar.markdown("""
    <div class="telemetry-card">
        <div class="telemetry-label">Active Corpus</div>
        <div class="telemetry-value">4 Verified PDFs</div>
    </div>
    <div class="telemetry-card">
        <div class="telemetry-label">Retrieval Architecture</div>
        <div class="telemetry-value">Chroma + BM25 (RRF)</div>
    </div>
    <div class="telemetry-card">
        <div class="telemetry-label">Embedding Model</div>
        <div class="telemetry-value">text-embedding-3-small</div>
    </div>
    <div class="telemetry-card">
        <div class="telemetry-label">Compliance Guardrails</div>
        <div class="telemetry-value">Strict Zero-Hallucination</div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Quick Test Prompts")
st.sidebar.markdown("Click any prompt below to evaluate the system instantly:")

preset_prompts = [
    "What are the core protocols for checking electrical voltage drops and hazards?",
    "What are the standard guidelines for writing objective safety citations?",
    "How should workplace safety hazards and inspection procedures be documented?",
    "What are the employer's responsibilities for maintaining general workplace safety?",
    "What steps are required when investigating an electrical or equipment hazard?"
]

if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = None

for prompt in preset_prompts:
    if st.sidebar.button(prompt):
        st.session_state.selected_prompt = prompt

# Initialize persistent chat history and follow-ups state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dynamic_follow_ups" not in st.session_state:
    st.session_state.dynamic_follow_ups = []

# Professional Tabbed Interface
tab1, tab2 = st.tabs(["💬 Compliance Copilot", "📂 Handbook Corpus Explorer"])

with tab1:
    st.markdown("### 💬 Regulatory Assistant")
    st.markdown("Query safety handbooks, inspection procedures, and compliance guidelines with verified source citations.")

    # Display past chat messages, citations, and interactive follow-up buttons
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                if "docs" in message:
                    with st.expander("📚 View Retrieved Source Citations & Metadata"):
                        for i, doc in enumerate(message["docs"]):
                            source_file = doc.metadata.get("source_file", "Unknown Source")
                            st.markdown(f"**Chunk {i+1}** | *Source: {source_file}*")
                            st.text(doc.page_content)
                            st.divider()
                
                # Show dynamic follow-up buttons under the latest assistant response
                if idx == len(st.session_state.messages) - 1 and st.session_state.dynamic_follow_ups:
                    st.markdown("**💡 Suggested Follow-ups:**")
                    cols = st.columns(len(st.session_state.dynamic_follow_ups))
                    for f_idx, fu_text in enumerate(st.session_state.dynamic_follow_ups):
                        with cols[f_idx]:
                            if st.button(fu_text, key=f"fu_{idx}_{f_idx}"):
                                st.session_state.selected_prompt = fu_text
                                st.rerun()

    # User input handling (chat input or sidebar/dynamic chip)
    user_input = st.chat_input("Ask a safety or compliance question...")
    query_to_run = user_input if user_input else st.session_state.selected_prompt

    if st.session_state.selected_prompt:
        st.session_state.selected_prompt = None

    if query_to_run:
        st.session_state.messages.append({"role": "user", "content": query_to_run})
        with st.chat_message("user"):
            st.write(query_to_run)
        
        with st.spinner("Retrieving verified handbook sections..."):
            retrieved_docs = retriever.search(query_to_run)
            context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert safety compliance assistant. Answer accurately using ONLY the provided context. Always cite the specific handbook source file."),
                ("human", "Context:\n{context}\n\nQuestion: {question}")
            ])
            
            chain = prompt_template | llm | StrOutputParser()
            answer = chain.invoke({"context": context_str, "question": query_to_run})
            
            # FIXED: Pass the context along with question and answer to generate deeply contextual follow-ups
            fu_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given the retrieved regulatory context, user question, and assistant answer, generate exactly 2 short, distinct follow-up questions that drill deeper into specific safety protocols or regulations found in the text. Return them separated by a newline, with no extra text or numbering."),
                ("human", "Context:\n{context}\n\nQuestion: {q}\nAnswer: {a}")
            ])
            fu_chain = fu_prompt | llm | StrOutputParser()
            fu_raw = fu_chain.invoke({"context": context_str, "q": query_to_run, "a": answer})
            st.session_state.dynamic_follow_ups = [line.strip() for line in fu_raw.split("\n") if line.strip()][:2]
            
        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("📚 View Retrieved Source Citations & Metadata"):
                for i, doc in enumerate(retrieved_docs):
                    source_file = doc.metadata.get("source_file", "Unknown Source")
                    st.markdown(f"**Chunk {i+1}** | *Source: {source_file}*")
                    st.text(doc.page_content)
                    st.divider()
                    
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "docs": retrieved_docs
        })
        
        st.rerun()

with tab2:
    st.subheader("Indexed Regulatory Source Documents")
    st.markdown("The compliance copilot is strictly grounded in the following **4 verified enterprise safety handbooks**:")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("**PH25-I-2**\n\nCitation and Order Writing Handbook")
        st.info("**OSHA 2209**\n\nSmall Business Safety and Health Handbook")
    with col_b:
        st.info("**PH26-V-1**\n\nElectrical Inspection Procedures Handbook")
        st.info("**OSHA 3885**\n\nSafety and Health Programs: Recommended Practices")
        
    st.markdown("---")
    st.markdown("### 🔍 Document Chunk Inspection Preview")
    st.markdown("Inspect raw vector store contents to verify chunk boundaries and metadata injection.")
    
    if st.button("Load Sample Chunks from Vector Store"):
        with st.spinner("Accessing vector store index..."):
            chunks = get_or_create_chunks()
            for i in range(min(3, len(chunks))):
                st.markdown(f"**Sample Chunk {i+1}** | *Source:* `{chunks[i].metadata.get('source_file')}`")
                st.text(chunks[i].page_content[:400] + "...")
                st.divider()
