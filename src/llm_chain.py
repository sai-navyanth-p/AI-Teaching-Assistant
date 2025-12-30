"""
LLM chain module for grounded RAG responses with citations.
"""
import logging
from typing import List, Dict, Any, Optional, Generator

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from .retriever import CourseRetriever
import config

logger = logging.getLogger(__name__)


# Grounded system prompt that enforces citation and prevents hallucination
GROUNDED_SYSTEM_PROMPT = """You are a helpful and knowledgeable Course Assistant for students. Your role is to answer questions about course materials STRICTLY based on the provided context from uploaded documents.

## CRITICAL RULES - YOU MUST FOLLOW THESE:

1. **ONLY use information from the provided context** to answer questions about course content, schedules, policies, assignments, exams, or any course-specific information.

2. **ALWAYS cite your sources** when providing factual information from the documents. Use this format: [Source: filename, Page X] for each fact.

3. **If information is NOT in the provided context**, you MUST explicitly say: "I don't have information about that in the uploaded course documents." NEVER make up or guess information.

4. **DO NOT hallucinate** course names, TA names, professor names, dates, deadlines, policies, or any specific details not present in the context.

5. **For general explanations** of concepts mentioned in course materials, you may provide educational explanations while citing which document discusses the topic.

6. **Be student-friendly**: Use clear language, organize responses well, and be helpful and encouraging.

7. **When asked about content from a specific document type** (syllabus, lecture, assignment, etc.), focus on those sources if available.

## Response Format:
- Provide a clear, well-structured answer
- Include inline citations [Source: filename, Page X] for specific facts
- If partially answering, note what information is missing
- Keep responses concise but complete

## Current Context from Course Documents:
{context}

Remember: You are grounded to these documents. If it's not in the context, say so clearly."""


class RAGChain:
    """
    RAG chain for generating grounded responses with citations.
    """
    
    def __init__(
        self,
        retriever: CourseRetriever,
        model_name: str = config.LLM_MODEL,
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS
    ):
        self.retriever = retriever
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", GROUNDED_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Create chain
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        logger.info(f"RAGChain initialized with model: {model_name}")
    
    def _format_chat_history(
        self,
        history: List[Dict[str, str]],
        max_turns: int = config.MAX_CHAT_HISTORY
    ) -> List:
        """
        Format chat history for the prompt.
        
        Args:
            history: List of dicts with 'role' and 'content'
            max_turns: Maximum number of turns to include
            
        Returns:
            List of message objects
        """
        messages = []
        
        # Take only recent history
        recent_history = history[-max_turns * 2:] if len(history) > max_turns * 2 else history
        
        for msg in recent_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'user':
                messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                messages.append(AIMessage(content=content))
        
        return messages
    
    def generate_response(
        self,
        question: str,
        course_id: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a grounded response with citations.
        
        Args:
            question: User's question
            course_id: Course ID for filtering
            chat_history: Previous conversation turns
            
        Returns:
            Dict with 'answer', 'sources', and 'evidence'
        """
        # Retrieve relevant documents
        documents = self.retriever.retrieve(
            query=question,
            course_id=course_id
        )
        
        # Format context
        context = self.retriever.format_context(documents)
        
        # Get source citations
        sources = self.retriever.get_source_citations(documents)
        
        # Format chat history
        formatted_history = self._format_chat_history(chat_history or [])
        
        # Generate response
        try:
            response = self.chain.invoke({
                "context": context,
                "chat_history": formatted_history,
                "question": question
            })
            
            return {
                "answer": response,
                "sources": sources,
                "evidence": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ],
                "num_sources": len(documents)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def generate_response_stream(
        self,
        question: str,
        course_id: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate a streaming response with citations.
        
        Args:
            question: User's question
            course_id: Course ID for filtering
            chat_history: Previous conversation turns
            
        Yields:
            Dicts with 'chunk' (text chunk), 'sources', 'evidence', 'done' (bool)
        """
        # Retrieve relevant documents
        documents = self.retriever.retrieve(
            query=question,
            course_id=course_id
        )
        
        # Format context
        context = self.retriever.format_context(documents)
        
        # Get source citations
        sources = self.retriever.get_source_citations(documents)
        
        # Format chat history
        formatted_history = self._format_chat_history(chat_history or [])
        
        # Evidence for display
        evidence = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
        
        # Yield sources and evidence first
        yield {
            "chunk": "",
            "sources": sources,
            "evidence": evidence,
            "num_sources": len(documents),
            "done": False
        }
        
        # Stream the response
        try:
            for chunk in self.chain.stream({
                "context": context,
                "chat_history": formatted_history,
                "question": question
            }):
                yield {
                    "chunk": chunk,
                    "sources": None,
                    "evidence": None,
                    "done": False
                }
            
            # Signal completion
            yield {
                "chunk": "",
                "sources": sources,
                "evidence": evidence,
                "done": True
            }
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield {
                "chunk": f"\n\n⚠️ Error generating response: {str(e)}",
                "sources": sources,
                "evidence": evidence,
                "done": True
            }
    
    def check_relevance(
        self,
        question: str,
        course_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if there are relevant documents for a question.
        
        Args:
            question: User's question
            course_id: Course ID for filtering
            
        Returns:
            Dict with relevance information
        """
        results = self.retriever.retrieve_with_scores(
            query=question,
            course_id=course_id
        )
        
        if not results:
            return {
                "has_relevant_docs": False,
                "message": "No documents found matching your query.",
                "top_score": 0.0
            }
        
        top_score = results[0][1] if results else 0.0
        
        return {
            "has_relevant_docs": len(results) > 0,
            "num_relevant": len(results),
            "top_score": top_score,
            "message": None
        }

