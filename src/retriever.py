"""
Retriever module for course-aware document retrieval with metadata filtering.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document

from .vector_store import VectorStoreManager
import config

logger = logging.getLogger(__name__)


class CourseRetriever:
    """
    Handles intelligent document retrieval with course awareness and metadata filtering.
    """
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        logger.info("CourseRetriever initialized")
    
    def detect_course_from_query(
        self,
        query: str,
        available_courses: List[str]
    ) -> Tuple[Optional[str], float]:
        """
        Attempt to detect which course the query is about.
        
        Only detects if the course ID is explicitly mentioned in the query.
        Otherwise, returns None to search across all courses and let
        semantic similarity find the most relevant documents.
        
        Args:
            query: User query string
            available_courses: List of available course IDs
            
        Returns:
            Tuple of (detected_course_id or None, confidence score 0-1)
        """
        if not available_courses:
            return None, 0.0
        
        query_lower = query.lower()
        
        # Look for explicit course ID mentions only
        for course in available_courses:
            course_patterns = [
                rf'\b{re.escape(course.lower())}\b',
                rf'\b{re.escape(course.lower().replace("-", " "))}\b',
                rf'\b{re.escape(course.lower().replace("_", " "))}\b'
            ]
            
            for pattern in course_patterns:
                if re.search(pattern, query_lower):
                    logger.info(f"Detected course {course} from explicit mention in query")
                    return course, 0.9
        
        # No explicit course mention - will search across all courses
        # Let semantic similarity find the most relevant documents
        logger.info("No explicit course mention - searching all courses")
        return None, 0.0
    
    def retrieve(
        self,
        query: str,
        course_id: Optional[str] = None,
        k: int = config.TOP_K_RESULTS,
        doc_type: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            course_id: Course ID to filter by (None for AUTO mode)
            k: Number of documents to retrieve
            doc_type: Optional document type filter
            
        Returns:
            List of relevant documents
        """
        filter_dict = {}
        
        if doc_type:
            filter_dict['doc_type'] = doc_type
        
        # Handle AUTO mode
        effective_course_id = course_id
        if course_id == config.AUTO_COURSE_ID or course_id is None:
            available_courses = self.vector_store.get_all_courses()
            detected_course, confidence = self.detect_course_from_query(query, available_courses)
            
            if detected_course and confidence >= 0.5:
                effective_course_id = detected_course
                logger.info(f"AUTO mode: Using detected course {detected_course} (confidence: {confidence})")
            else:
                # Search across all courses
                effective_course_id = None
                logger.info("AUTO mode: Searching across all courses")
        
        documents = self.vector_store.similarity_search(
            query=query,
            course_id=effective_course_id,
            k=k,
            filter_dict=filter_dict if filter_dict else None
        )
        
        return documents
    
    def retrieve_with_scores(
        self,
        query: str,
        course_id: Optional[str] = None,
        k: int = config.TOP_K_RESULTS,
        score_threshold: float = config.SIMILARITY_THRESHOLD
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with relevance scores, filtering by threshold.
        
        Args:
            query: User query
            course_id: Course ID to filter by
            k: Number of documents to retrieve
            score_threshold: Minimum relevance score
            
        Returns:
            List of (Document, score) tuples
        """
        effective_course_id = course_id
        if course_id == config.AUTO_COURSE_ID or course_id is None:
            available_courses = self.vector_store.get_all_courses()
            detected_course, confidence = self.detect_course_from_query(query, available_courses)
            
            if detected_course and confidence >= 0.5:
                effective_course_id = detected_course
            else:
                effective_course_id = None
        
        results = self.vector_store.similarity_search_with_scores(
            query=query,
            course_id=effective_course_id,
            k=k
        )
        
        # Filter by score threshold
        filtered_results = [
            (doc, score) for doc, score in results
            if score >= score_threshold
        ]
        
        return filtered_results
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string for the LLM.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string with source attribution
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            source = metadata.get('source_file', 'Unknown')
            page = metadata.get('page_number', 'N/A')
            doc_type = metadata.get('doc_type', 'unknown')
            course = metadata.get('course_id', 'Unknown')
            
            # Create source citation
            citation = f"[Source: {source}"
            if page != 'N/A':
                citation += f", Page {page}"
            citation += f", Type: {doc_type}, Course: {course}]"
            
            context_parts.append(f"--- Document {i} {citation} ---\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def get_source_citations(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract source citations from documents for display.
        
        Args:
            documents: List of documents
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        seen = set()
        
        for doc in documents:
            metadata = doc.metadata
            source = metadata.get('source_file', 'Unknown')
            page = metadata.get('page_number', 'N/A')
            
            # Create unique key
            key = f"{source}_{page}"
            if key not in seen:
                seen.add(key)
                citations.append({
                    'source_file': source,
                    'page_number': page,
                    'doc_type': metadata.get('doc_type', 'unknown'),
                    'course_id': metadata.get('course_id', 'Unknown'),
                    'snippet': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
        
        return citations

