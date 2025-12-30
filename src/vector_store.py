"""
Vector store manager for ChromaDB operations.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


def get_embedding_model():
    """
    Get the appropriate embedding model based on config.
    
    Returns either local HuggingFace embeddings or OpenAI embeddings
    based on the EMBEDDING_PROVIDER setting in config.
    """
    if config.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        model_name = config.OPENAI_EMBEDDING_MODEL
        logger.info(f"Using OpenAI embeddings: {model_name}")
        return OpenAIEmbeddings(model=model_name)
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_name = config.LOCAL_EMBEDDING_MODEL
        logger.info(f"Using local HuggingFace embeddings: {model_name}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )


class VectorStoreManager:
    """Manages ChromaDB vector store operations with persistent storage."""
    
    def __init__(
        self,
        persist_directory: Path = config.CHROMA_PERSIST_DIR,
        collection_name: str = config.CHROMA_COLLECTION_NAME
    ):
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        
        # Initialize embeddings based on config (local or OpenAI)
        self.embeddings = get_embedding_model()
        
        # Initialize ChromaDB client with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize or get the vector store
        self._init_vector_store()
        
        logger.info(f"VectorStoreManager initialized with collection: {collection_name}")
    
    def _init_vector_store(self):
        """Initialize or load the existing vector store."""
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects with metadata
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents provided to add")
            return []
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Generate unique IDs based on chunk_id in metadata
        ids = [doc.metadata.get('chunk_id', str(i)) for i, doc in enumerate(documents)]
        
        # Add documents
        self.vector_store.add_documents(documents=documents, ids=ids)
        
        logger.info(f"Successfully added {len(documents)} documents")
        return ids
    
    def similarity_search(
        self,
        query: str,
        course_id: Optional[str] = None,
        k: int = config.TOP_K_RESULTS,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search with optional metadata filtering.
        
        Args:
            query: Search query string
            course_id: Optional course ID to filter by
            k: Number of results to return
            filter_dict: Additional filter criteria
            
        Returns:
            List of relevant Document objects
        """
        # Build filter
        where_filter = {}
        
        if course_id and course_id != config.AUTO_COURSE_ID:
            where_filter["course_id"] = course_id.upper()
        
        if filter_dict:
            where_filter.update(filter_dict)
        
        logger.info(f"Searching for: '{query[:50]}...' with filter: {where_filter}")
        
        try:
            if where_filter:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=where_filter
                )
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k
                )
            
            logger.info(f"Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def similarity_search_with_scores(
        self,
        query: str,
        course_id: Optional[str] = None,
        k: int = config.TOP_K_RESULTS,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search and return documents with relevance scores.
        
        Args:
            query: Search query string
            course_id: Optional course ID to filter by
            k: Number of results to return
            filter_dict: Additional filter criteria
            
        Returns:
            List of tuples (Document, score)
        """
        where_filter = {}
        
        if course_id and course_id != config.AUTO_COURSE_ID:
            where_filter["course_id"] = course_id.upper()
        
        if filter_dict:
            where_filter.update(filter_dict)
        
        try:
            if where_filter:
                results = self.vector_store.similarity_search_with_relevance_scores(
                    query=query,
                    k=k,
                    filter=where_filter
                )
            else:
                results = self.vector_store.similarity_search_with_relevance_scores(
                    query=query,
                    k=k
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search with scores: {str(e)}")
            return []
    
    def get_all_courses(self) -> List[str]:
        """Get all unique course IDs in the vector store."""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            
            # Get all metadata
            results = collection.get(include=['metadatas'])
            
            if results and results['metadatas']:
                courses = set()
                for metadata in results['metadatas']:
                    if metadata and 'course_id' in metadata:
                        courses.add(metadata['course_id'])
                return sorted(list(courses))
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting courses: {str(e)}")
            return []
    
    def get_documents_by_course(self, course_id: str) -> List[Dict[str, Any]]:
        """
        Get all unique documents (by source file) for a course.
        
        Args:
            course_id: Course identifier
            
        Returns:
            List of document info dicts with file name, type, upload time
        """
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            
            results = collection.get(
                where={"course_id": course_id.upper()},
                include=['metadatas']
            )
            
            if results and results['metadatas']:
                # Group by source file
                docs_map = {}
                for metadata in results['metadatas']:
                    if metadata:
                        source = metadata.get('source_file', 'Unknown')
                        if source not in docs_map:
                            docs_map[source] = {
                                'source_file': source,
                                'doc_type': metadata.get('doc_type', 'unknown'),
                                'upload_timestamp': metadata.get('upload_timestamp', ''),
                                'file_type': metadata.get('file_type', 'unknown'),
                                'total_pages': metadata.get('total_pages', 1),
                                'course_id': metadata.get('course_id', '')
                            }
                
                return list(docs_map.values())
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting documents for course {course_id}: {str(e)}")
            return []
    
    def delete_document(self, course_id: str, source_file: str) -> bool:
        """
        Delete all chunks of a document from the vector store.
        
        Args:
            course_id: Course identifier
            source_file: Name of the source file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            
            # Get IDs of chunks to delete
            results = collection.get(
                where={
                    "$and": [
                        {"course_id": course_id.upper()},
                        {"source_file": source_file}
                    ]
                },
                include=['metadatas']
            )
            
            if results and results['ids']:
                collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for {source_file} from course {course_id}")
                return True
            
            logger.warning(f"No chunks found for {source_file} in course {course_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error deleting document {source_file}: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                'total_chunks': count,
                'collection_name': self.collection_name,
                'courses': self.get_all_courses()
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                'total_chunks': 0,
                'collection_name': self.collection_name,
                'courses': []
            }

