# 🎓 Multi-Course AI Teaching Assistant (RAG)

A local-first RAG (Retrieval-Augmented Generation) application that enables students to upload course documents and have grounded, citation-backed conversations with an AI assistant. 

Upload your course materials, ask questions in plain English, and get accurate answers with source citations.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4+-purple.svg)

## ✨ Features

### Core Capabilities
- **Multi-Course Support**: Upload and organize documents by course with strict separation
- **Smart Document Processing**: Automatic chunking with rich metadata for PDFs and TXT files
- **Grounded Responses**: Every answer is strictly based on uploaded documents with citations
- **Citation System**: File name and page number citations for every factual claim
- **Evidence Display**: Expandable sections showing exact quotes from source documents
- **Course Auto-Detection**: System can detect relevant course from question context

### Document Management
- Upload PDFs and TXT files (lectures, assignments, syllabi, exams, schedules)
- Automatic indexing immediately after upload
- Delete individual documents from the vector store
- View all documents organized by course

### User Interface
- Clean, elegant light-themed Streamlit UI
- Streaming responses for real-time feedback
- Persistent chat history within session
- Course filtering via sidebar selector
- Source and evidence sections for transparency

## 🏗️ Architecture

```
AI-Teaching-Assistant/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env                        # Environment variables (create this)
├── src/
│   ├── __init__.py
│   ├── document_processor.py   # PDF/TXT loading and chunking
│   ├── vector_store.py         # ChromaDB operations
│   ├── retriever.py            # Course-aware retrieval
│   ├── llm_chain.py            # LangChain RAG chain
│   └── utils.py                # Helper utilities
├── data/
│   └── chroma_db/              # Persistent vector store
└── logs/
    └── rag_assistant.log       # Application logs
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- OpenAI API key

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd AI-Teaching-Assistant
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   ```
   
   Or export directly:
   ```bash
   export OPENAI_API_KEY=your-openai-api-key-here
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage Guide

### Uploading Documents

1. In the sidebar, enter a **Course ID** (e.g., "CS101", "MATH201")
2. Select the **Document Type** (lecture, assignment, syllabus, exam, schedule, misc)
3. Upload one or more **PDF or TXT files**
4. Click **Upload & Index** to process and store the documents

### Asking Questions

1. Select a course from the **Course Selection** dropdown, or use **AUTO** mode
2. Type your question in the chat input at the bottom
3. The assistant will search your documents and provide a grounded answer
4. Check the **Sources** section to see which documents were used
5. Expand **View Evidence** to see exact quotes from the documents

### Course Modes

- **AUTO**: The system attempts to detect the relevant course from your question
- **Specific Course**: Questions are filtered to only search that course's documents
- **MISC**: For miscellaneous documents not associated with a specific course

### Example Questions

- "What is the grading policy for this course?"
- "When is the midterm exam?"
- "Who is the TA for office hours?"
- "Explain the concept of recursion from the lecture notes"
- "What topics are covered in week 3?"
- "Compare the requirements for Assignment 1 and Assignment 2"

## 🔒 How Grounded Answers Work

This application implements several mechanisms to ensure responses are factually grounded in uploaded documents:

### 1. Retrieval-Augmented Generation (RAG)
Every user question triggers a similarity search against the vector store. Only content from uploaded documents is included in the LLM's context window.

### 2. Strict System Prompt
The LLM receives explicit instructions to:
- Only use information from the provided context
- Never fabricate citations or information
- Explicitly state when information is not found
- Include source citations for all factual claims

### 3. Metadata-Based Filtering
Documents are tagged with rich metadata (course_id, doc_type, source_file, page_number) enabling precise retrieval filtering.

### 4. Citation Enforcement
The system prompt requires the LLM to cite sources using `[Source: filename, Page X]` format for every piece of information.

### 5. Evidence Display
Users can verify answers by viewing the exact text chunks that were retrieved and used to generate the response.

### 6. No Hallucination Policy
When relevant information isn't found, the system explicitly states: *"I don't have information about that in the uploaded course documents."*

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Chunking settings
CHUNK_SIZE = 1000        # Characters per chunk
CHUNK_OVERLAP = 200      # Overlap between chunks

# Retrieval settings
TOP_K_RESULTS = 5        # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.3

# LLM settings
LLM_MODEL = "gpt-4o-mini"  # OpenAI model to use
LLM_TEMPERATURE = 0.1      # Lower = more deterministic
LLM_MAX_TOKENS = 1500

# Embedding model (local)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

## 🛠️ Technical Details

### Vector Store
- **ChromaDB** with persistent on-disk storage
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for local embeddings
- Single collection with metadata-based course separation

### Chunk Metadata
Each chunk stores:
- `course_id`: Course identifier
- `doc_type`: Document type (lecture, assignment, etc.)
- `source_file`: Original filename
- `page_number`: Page number (for PDFs)
- `chunk_id`: Unique chunk identifier
- `upload_timestamp`: When the document was indexed
- `file_type`: File extension (pdf, txt)

### Document Processing
- **PDFs**: Extracted using `pdfplumber` with page-level processing
- **TXT**: UTF-8 decoded with graceful error handling
- **Chunking**: Recursive character splitting with configurable size/overlap

## 🔧 Troubleshooting

### Common Issues

1. **"OpenAI API Key Required" warning**
   - Ensure your `.env` file exists and contains a valid API key
   - Try restarting the Streamlit app after adding the key

2. **Slow document processing**
   - Large PDFs take time to process; this is normal
   - Check logs in `logs/rag_assistant.log` for progress

3. **"No documents found" errors**
   - Verify documents were successfully uploaded (check sidebar document list)
   - Ensure you're querying the correct course

4. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Verify Python version is 3.9+

### Logs
Application logs are stored in `logs/rag_assistant.log` and include:
- Document processing events
- Retrieval queries and results
- Error messages with stack traces

## 📋 Roadmap

Potential future enhancements:
- [ ] Support for additional file formats (PPTX, tex, etc)
- [ ] Export chat history
- [ ] Multi-user support with authentication
- [ ] Advanced search filters (date range, doc type)

## 📄 License

This project is open source and available under the MIT License.

---

