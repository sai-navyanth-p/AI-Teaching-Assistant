"""
Document processor for loading and chunking PDF and TXT files.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, BinaryIO
import hashlib
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config

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


class DocumentProcessor:
    """Handles document loading, parsing, and chunking with rich metadata."""
    
    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def process_pdf(
        self,
        file_content: bytes,
        filename: str,
        course_id: str,
        doc_type: str
    ) -> List[Document]:
        """
        Process a PDF file and return chunked documents with metadata.
        
        Args:
            file_content: Raw bytes of the PDF file
            filename: Original filename
            course_id: Course identifier
            doc_type: Type of document (lecture, assignment, etc.)
            
        Returns:
            List of Document objects with rich metadata
        """
        import pdfplumber
        import io
        
        logger.info(f"Processing PDF: {filename} for course {course_id}")
        
        documents = []
        upload_timestamp = datetime.now().isoformat()
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        
        try:
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    
                    if not text.strip():
                        logger.debug(f"Page {page_num} of {filename} is empty, skipping")
                        continue
                    
                    # Create chunks for this page
                    chunks = self.text_splitter.split_text(text)
                    
                    for chunk_idx, chunk_text in enumerate(chunks):
                        chunk_id = f"{file_hash}_{page_num}_{chunk_idx}"
                        
                        doc = Document(
                            page_content=chunk_text,
                            metadata={
                                "course_id": course_id.upper(),
                                "doc_type": doc_type,
                                "source_file": filename,
                                "page_number": page_num,
                                "chunk_id": chunk_id,
                                "chunk_index": chunk_idx,
                                "upload_timestamp": upload_timestamp,
                                "file_type": "pdf",
                                "total_pages": len(pdf.pages)
                            }
                        )
                        documents.append(doc)
            
            logger.info(f"Successfully processed PDF {filename}: {len(documents)} chunks created")
            
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {str(e)}")
            raise ValueError(f"Failed to process PDF '{filename}': {str(e)}")
        
        return documents
    
    def process_txt(
        self,
        file_content: bytes,
        filename: str,
        course_id: str,
        doc_type: str
    ) -> List[Document]:
        """
        Process a TXT file and return chunked documents with metadata.
        
        Args:
            file_content: Raw bytes of the TXT file
            filename: Original filename
            course_id: Course identifier
            doc_type: Type of document (lecture, assignment, etc.)
            
        Returns:
            List of Document objects with rich metadata
        """
        logger.info(f"Processing TXT: {filename} for course {course_id}")
        
        documents = []
        upload_timestamp = datetime.now().isoformat()
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        
        try:
            # Decode text content
            text = file_content.decode('utf-8', errors='ignore')
            
            if not text.strip():
                logger.warning(f"TXT file {filename} is empty")
                return documents
            
            # Create chunks
            chunks = self.text_splitter.split_text(text)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_id = f"{file_hash}_1_{chunk_idx}"
                
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "course_id": course_id.upper(),
                        "doc_type": doc_type,
                        "source_file": filename,
                        "page_number": 1,  # TXT files don't have pages
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_idx,
                        "upload_timestamp": upload_timestamp,
                        "file_type": "txt",
                        "total_pages": 1
                    }
                )
                documents.append(doc)
            
            logger.info(f"Successfully processed TXT {filename}: {len(documents)} chunks created")
            
        except Exception as e:
            logger.error(f"Error processing TXT {filename}: {str(e)}")
            raise ValueError(f"Failed to process TXT '{filename}': {str(e)}")
        
        return documents
    
    def process_file(
        self,
        file_content: bytes,
        filename: str,
        course_id: str,
        doc_type: str
    ) -> List[Document]:
        """
        Process a file based on its extension.
        
        Args:
            file_content: Raw bytes of the file
            filename: Original filename
            course_id: Course identifier
            doc_type: Type of document
            
        Returns:
            List of Document objects with rich metadata
        """
        extension = Path(filename).suffix.lower()
        
        if extension == '.pdf':
            return self.process_pdf(file_content, filename, course_id, doc_type)
        elif extension in ['.txt', '.text']:
            return self.process_txt(file_content, filename, course_id, doc_type)
        else:
            raise ValueError(f"Unsupported file type: {extension}. Supported types: .pdf, .txt")
    
    def process_multiple_files(
        self,
        files: List[Dict[str, Any]],
        course_id: str,
        doc_type: str
    ) -> tuple[List[Document], List[str]]:
        """
        Process multiple files and return all chunked documents.
        
        Args:
            files: List of dicts with 'content' (bytes) and 'name' (str)
            course_id: Course identifier
            doc_type: Type of documents
            
        Returns:
            Tuple of (list of documents, list of error messages)
        """
        all_documents = []
        errors = []
        
        for file_info in files:
            try:
                docs = self.process_file(
                    file_content=file_info['content'],
                    filename=file_info['name'],
                    course_id=course_id,
                    doc_type=doc_type
                )
                all_documents.extend(docs)
            except Exception as e:
                error_msg = f"Error processing {file_info['name']}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        return all_documents, errors

