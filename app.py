"""
Multi-Course AI Teaching Assistant (RAG) - Streamlit Application

A local-first RAG application for students to upload course documents
and have grounded conversations with citations.
"""
import streamlit as st
from datetime import datetime
import logging
from typing import Optional
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.retriever import CourseRetriever
from src.llm_chain import RAGChain
from src.utils import (
    sanitize_course_id,
    format_timestamp,
    get_doc_type_icon,
    get_file_icon,
    validate_course_id
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elegant light theme
st.markdown("""
<style>
    /* Import elegant fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root variables */
    :root {
        --primary: #2563eb;
        --primary-light: #3b82f6;
        --primary-dark: #1d4ed8;
        --accent: #f59e0b;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --bg-primary: #fafbfc;
        --bg-secondary: #ffffff;
        --bg-tertiary: #f1f5f9;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --border: #e2e8f0;
        --shadow: rgba(15, 23, 42, 0.08);
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #f0f4ff 100%);
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 1.75rem;
        font-weight: 600;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.95rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* Chat container */
    .chat-container {
        background: var(--bg-secondary);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px var(--shadow);
        border: 1px solid var(--border);
        margin-bottom: 1rem;
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.75rem 0;
        margin-left: 15%;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.15);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .assistant-message {
        background: var(--bg-tertiary);
        color: var(--text-primary);
        padding: 1rem 1.25rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.75rem 0;
        margin-right: 15%;
        border: 1px solid var(--border);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Sources card */
    .sources-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.75rem;
        border-left: 4px solid var(--accent);
    }
    
    .sources-card h4 {
        color: #92400e;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .source-item {
        background: rgba(255, 255, 255, 0.7);
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    
    /* Evidence expander */
    .evidence-card {
        background: var(--bg-tertiary);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.5rem;
        border: 1px solid var(--border);
    }
    
    .evidence-snippet {
        background: white;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.8rem;
        color: var(--text-secondary);
        border-left: 3px solid var(--primary-light);
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.5;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stFileUploader label {
        color: var(--text-primary);
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #a7f3d0;
    }
    
    .upload-section h4 {
        color: #065f46;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0 0 0.75rem 0;
    }
    
    /* Document list */
    .doc-list-item {
        background: white;
        border-radius: 10px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.2s ease;
    }
    
    .doc-list-item:hover {
        border-color: var(--primary-light);
        box-shadow: 0 2px 8px var(--shadow);
    }
    
    .doc-info {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .doc-name {
        font-weight: 500;
        color: var(--text-primary);
        font-size: 0.85rem;
    }
    
    .doc-meta {
        font-size: 0.75rem;
        color: var(--text-muted);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.25rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Stats card */
    .stats-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid var(--border);
        margin: 0.5rem 0;
    }
    
    .stats-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    .stats-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        border-radius: 10px;
        border: 1px solid var(--border);
        font-family: 'Outfit', sans-serif;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    /* Chat input */
    .stChatInput {
        border-radius: 16px;
    }
    
    .stChatInput > div {
        border-radius: 16px;
        border: 2px solid var(--border);
        background: white;
    }
    
    .stChatInput > div:focus-within {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--bg-tertiary);
        border-radius: 10px;
        font-weight: 500;
        color: var(--text-primary);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: var(--border);
        margin: 1.5rem 0;
    }
    
    /* Success/Warning messages */
    .stSuccess {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 10px;
    }
    
    .stWarning {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
    }
    
    .stError {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 10px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-tertiary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--text-muted);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStoreManager()
    
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = CourseRetriever(st.session_state.vector_store)
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = RAGChain(st.session_state.retriever)
    
    if "selected_course" not in st.session_state:
        st.session_state.selected_course = config.AUTO_COURSE_ID
    
    if "available_courses" not in st.session_state:
        st.session_state.available_courses = []


def refresh_courses():
    """Refresh the list of available courses."""
    st.session_state.available_courses = st.session_state.vector_store.get_all_courses()


def render_sidebar():
    """Render the sidebar with course selection and document management."""
    with st.sidebar:
        st.markdown("### 🎓 AI Teaching Assistant")
        st.markdown("---")
        
        # Course selector
        st.markdown("#### 📚 Course Selection")
        
        refresh_courses()
        course_options = [config.AUTO_COURSE_ID] + st.session_state.available_courses + [config.MISC_COURSE_ID]
        course_options = list(dict.fromkeys(course_options))  # Remove duplicates while preserving order
        
        selected = st.selectbox(
            "Active Course",
            options=course_options,
            index=course_options.index(st.session_state.selected_course) if st.session_state.selected_course in course_options else 0,
            help="Select a course to filter questions, or use AUTO to let the system detect the course."
        )
        st.session_state.selected_course = selected
        
        if selected == config.AUTO_COURSE_ID:
            st.info("🔍 AUTO mode: The system will attempt to detect the relevant course from your question.")
        
        st.markdown("---")
        
        # Upload section
        st.markdown("#### 📤 Upload Documents")
        
        with st.container():
            # Course ID for upload
            upload_course_id = st.text_input(
                "Course ID",
                placeholder="e.g., CS101, MATH201",
                help="Enter the course ID for these documents"
            )
            
            # Document type selector
            doc_type = st.selectbox(
                "Document Type",
                options=config.DOCUMENT_TYPES,
                format_func=lambda x: f"{get_doc_type_icon(x)} {x.title()}"
            )
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload Files",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help="Upload PDF or TXT files"
            )
            
            # Upload button
            if st.button("📥 Upload & Index", use_container_width=True, type="primary"):
                if not upload_course_id:
                    st.error("Please enter a Course ID")
                elif not uploaded_files:
                    st.error("Please select files to upload")
                else:
                    is_valid, error_msg = validate_course_id(upload_course_id)
                    if not is_valid:
                        st.error(error_msg)
                    else:
                        with st.spinner("Processing documents..."):
                            process_uploads(uploaded_files, upload_course_id, doc_type)
        
        st.markdown("---")
        
        # Document list for selected course
        if selected and selected != config.AUTO_COURSE_ID:
            st.markdown(f"#### 📁 Documents in {selected}")
            
            docs = st.session_state.vector_store.get_documents_by_course(selected)
            
            if docs:
                for doc in docs:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        icon = get_file_icon(doc['file_type'])
                        type_icon = get_doc_type_icon(doc['doc_type'])
                        st.markdown(f"""
                        <div style="font-size: 0.85rem; margin-bottom: 0.5rem;">
                            {icon} <strong>{doc['source_file']}</strong><br>
                            <span style="color: #64748b; font-size: 0.75rem;">
                                {type_icon} {doc['doc_type'].title()} • {doc['total_pages']} page(s)
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['source_file']}", help="Delete document"):
                            if st.session_state.vector_store.delete_document(selected, doc['source_file']):
                                st.success(f"Deleted {doc['source_file']}")
                                st.rerun()
                            else:
                                st.error("Failed to delete")
            else:
                st.info("No documents uploaded for this course yet.")
        
        st.markdown("---")
        
        # Stats
        stats = st.session_state.vector_store.get_collection_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", stats['total_chunks'])
        with col2:
            st.metric("Courses", len(stats['courses']))
        
        # Reset conversation
        st.markdown("---")
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def process_uploads(uploaded_files, course_id: str, doc_type: str):
    """Process uploaded files and add to vector store."""
    sanitized_course = sanitize_course_id(course_id)
    
    files_data = []
    for file in uploaded_files:
        files_data.append({
            'content': file.read(),
            'name': file.name
        })
    
    documents, errors = st.session_state.doc_processor.process_multiple_files(
        files=files_data,
        course_id=sanitized_course,
        doc_type=doc_type
    )
    
    if documents:
        ids = st.session_state.vector_store.add_documents(documents)
        st.success(f"✅ Successfully indexed {len(documents)} chunks from {len(files_data) - len(errors)} file(s)")
        refresh_courses()
        
        # Update selected course to the newly uploaded one
        if st.session_state.selected_course == config.AUTO_COURSE_ID:
            st.session_state.selected_course = sanitized_course
    
    for error in errors:
        st.error(error)


def render_chat_message(role: str, content: str, sources: Optional[list] = None, evidence: Optional[list] = None):
    """Render a chat message with optional sources and evidence."""
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Render sources
        if sources:
            sources_html = "".join([
                f'<div class="source-item">📄 {s["source_file"]} (Page {s["page_number"]}) - {s["doc_type"].title()}</div>'
                for s in sources
            ])
            st.markdown(f"""
            <div class="sources-card">
                <h4>📚 Sources ({len(sources)})</h4>
                {sources_html}
            </div>
            """, unsafe_allow_html=True)
        
        # Render evidence in expander
        if evidence:
            with st.expander(f"📋 View Evidence ({len(evidence)} snippets)"):
                for i, e in enumerate(evidence, 1):
                    meta = e['metadata']
                    st.markdown(f"""
                    <div class="evidence-snippet">
                        <strong>Source {i}:</strong> {meta.get('source_file', 'Unknown')} (Page {meta.get('page_number', 'N/A')})<br><br>
                        {e['content'][:500]}{'...' if len(e['content']) > 500 else ''}
                    </div>
                    """, unsafe_allow_html=True)


def render_main_chat():
    """Render the main chat interface."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Multi-Course AI Teaching Assistant (RAG):  </h1>
        <p>Your Intelligent Study Companion. Ask questions, get cited answers from your course materials.</p>

    </div>
    """, unsafe_allow_html=True)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("""
        ⚠️ **OpenAI API Key Required**
        
        Please set your `OPENAI_API_KEY` environment variable or add it to a `.env` file in the project root.
        
        ```
        OPENAI_API_KEY=your-api-key-here
        ```
        """)
        return
    
    # Display current course context
    if st.session_state.selected_course != config.AUTO_COURSE_ID:
        st.info(f"📚 Currently filtering by: **{st.session_state.selected_course}**")
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            render_chat_message(
                role=msg["role"],
                content=msg["content"],
                sources=msg.get("sources"),
                evidence=msg.get("evidence")
            )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your course materials..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Render user message immediately
        render_chat_message("user", prompt)
        
        # Check if there are any documents
        stats = st.session_state.vector_store.get_collection_stats()
        if stats['total_chunks'] == 0:
            no_docs_msg = "📚 No documents have been uploaded yet. Please upload some course materials using the sidebar to get started."
            st.session_state.messages.append({
                "role": "assistant",
                "content": no_docs_msg
            })
            render_chat_message("assistant", no_docs_msg)
            st.rerun()
            return
        
        # Generate response with streaming
        with st.spinner("Searching documents and generating response..."):
            try:
                # Prepare chat history for context
                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]  # Exclude current message
                ]
                
                # Get course ID for filtering
                course_id = st.session_state.selected_course
                
                # Generate streaming response
                response_placeholder = st.empty()
                full_response = ""
                sources = None
                evidence = None
                
                for chunk_data in st.session_state.rag_chain.generate_response_stream(
                    question=prompt,
                    course_id=course_id,
                    chat_history=chat_history
                ):
                    if chunk_data["sources"] is not None:
                        sources = chunk_data["sources"]
                        evidence = chunk_data["evidence"]
                    
                    full_response += chunk_data["chunk"]
                    response_placeholder.markdown(f"""
                    <div class="assistant-message">
                        {full_response}▌
                    </div>
                    """, unsafe_allow_html=True)
                
                # Final render without cursor
                response_placeholder.empty()
                
                # Add to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                    "evidence": evidence
                })
                
                # Render final message with sources
                render_chat_message("assistant", full_response, sources, evidence)
                
            except Exception as e:
                error_msg = f"⚠️ Error generating response: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
        
        st.rerun()


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_main_chat()


if __name__ == "__main__":
    main()

