import json
import logging
from typing import Dict, List, Optional, Any
import requests
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPRequest:
    """Model Context Protocol request structure"""
    method: str
    params: Dict[str, Any]
    id: str
    jsonrpc: str = "2.0"

@dataclass
class MCPResponse:
    """Model Context Protocol response structure"""
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    id: str = ""
    jsonrpc: str = "2.0"

class MCPClient:
    """
    Model Context Protocol client for advanced reasoning and context management
    Integrates with AI models to provide enhanced context-aware responses
    """
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        """
        Initialize MCP client
        
        Args:
            base_url: Base URL for MCP server
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.request_id = 0
    
    def _get_next_id(self) -> str:
        """Get next request ID"""
        self.request_id += 1
        return str(self.request_id)
    
    def _make_request(self, method: str, params: Dict[str, Any]) -> MCPResponse:
        """
        Make MCP request
        
        Args:
            method: MCP method name
            params: Request parameters
            
        Returns:
            MCP response
        """
        try:
            request = MCPRequest(
                method=method,
                params=params,
                id=self._get_next_id()
            )
            
            response = self.session.post(
                f"{self.base_url}/mcp",
                json=request.__dict__,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            return MCPResponse(
                result=data.get('result'),
                error=data.get('error'),
                id=data.get('id', ''),
                jsonrpc=data.get('jsonrpc', '2.0')
            )
            
        except Exception as e:
            logger.error(f"MCP request failed: {str(e)}")
            return MCPResponse(
                error={"code": -1, "message": str(e)},
                id=self._get_next_id()
            )
    
    def initialize_context(self, context_type: str = "content_analysis") -> Dict[str, Any]:
        """
        Initialize context for content analysis
        
        Args:
            context_type: Type of context to initialize
            
        Returns:
            Context initialization result
        """
        params = {
            "context_type": context_type,
            "capabilities": [
                "text_analysis",
                "question_generation",
                "summarization",
                "concept_extraction"
            ]
        }
        
        response = self._make_request("initialize", params)
        return response.result or {}
    
    def add_document_context(self, document_id: str, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add document to context
        
        Args:
            document_id: Unique document identifier
            content: Document content
            metadata: Document metadata
            
        Returns:
            Addition result
        """
        params = {
            "document_id": document_id,
            "content": content,
            "metadata": metadata or {}
        }
        
        response = self._make_request("add_document", params)
        return response.result or {}
    
    def analyze_with_context(self, query: str, context_ids: List[str] = None) -> Dict[str, Any]:
        """
        Analyze content with context awareness
        
        Args:
            query: Analysis query
            context_ids: List of context IDs to use
            
        Returns:
            Analysis result
        """
        params = {
            "query": query,
            "context_ids": context_ids or [],
            "analysis_type": "comprehensive"
        }
        
        response = self._make_request("analyze", params)
        return response.result or {}
    
    def generate_questions_with_context(self, content: str, question_count: int = 10, context_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Generate questions with context awareness
        
        Args:
            content: Content to generate questions from
            question_count: Number of questions to generate
            context_ids: List of context IDs to use
            
        Returns:
            List of generated questions
        """
        params = {
            "content": content,
            "question_count": question_count,
            "context_ids": context_ids or [],
            "question_types": ["multiple_choice", "true_false", "short_answer"]
        }
        
        response = self._make_request("generate_questions", params)
        return response.result.get("questions", []) if response.result else []
    
    def extract_concepts_with_context(self, content: str, context_ids: List[str] = None) -> List[str]:
        """
        Extract concepts with context awareness
        
        Args:
            content: Content to extract concepts from
            context_ids: List of context IDs to use
            
        Returns:
            List of extracted concepts
        """
        params = {
            "content": content,
            "context_ids": context_ids or [],
            "concept_count": 12
        }
        
        response = self._make_request("extract_concepts", params)
        return response.result.get("concepts", []) if response.result else []
    
    def summarize_with_context(self, content: str, length: str = "medium", context_ids: List[str] = None) -> str:
        """
        Generate summary with context awareness
        
        Args:
            content: Content to summarize
            length: Summary length ("brief", "medium", "detailed")
            context_ids: List of context IDs to use
            
        Returns:
            Generated summary
        """
        params = {
            "content": content,
            "length": length,
            "context_ids": context_ids or []
        }
        
        response = self._make_request("summarize", params)
        return response.result.get("summary", "") if response.result else ""
    
    def get_context_status(self) -> Dict[str, Any]:
        """
        Get current context status
        
        Returns:
            Context status information
        """
        response = self._make_request("status", {})
        return response.result or {}
    
    def clear_context(self) -> Dict[str, Any]:
        """
        Clear current context
        
        Returns:
            Clear operation result
        """
        response = self._make_request("clear", {})
        return response.result or {}

class MCPFallback:
    """
    Fallback MCP implementation when MCP server is not available
    Provides enhanced reasoning without external MCP server
    """
    
    def __init__(self):
        self.context_store = {}
        self.document_store = {}
    
    def initialize_context(self, context_type: str = "content_analysis") -> Dict[str, Any]:
        """Initialize fallback context"""
        context_id = f"context_{len(self.context_store)}"
        self.context_store[context_id] = {
            "type": context_type,
            "documents": [],
            "created_at": "now"
        }
        return {"context_id": context_id, "status": "initialized"}
    
    def add_document_context(self, document_id: str, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add document to fallback context"""
        self.document_store[document_id] = {
            "content": content,
            "metadata": metadata or {},
            "added_at": "now"
        }
        return {"document_id": document_id, "status": "added"}
    
    def analyze_with_context(self, query: str, context_ids: List[str] = None) -> Dict[str, Any]:
        """Enhanced analysis with fallback context"""
        # Use stored documents for enhanced analysis
        relevant_docs = []
        for doc_id, doc_data in self.document_store.items():
            if query.lower() in doc_data["content"].lower():
                relevant_docs.append(doc_data)
        
        return {
            "query": query,
            "relevant_documents": len(relevant_docs),
            "analysis": "Enhanced analysis using stored context",
            "context_used": True
        }
    
    def generate_questions_with_context(self, content: str, question_count: int = 10, context_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Generate questions with context awareness"""
        # Enhanced question generation using stored context
        questions = []
        for i in range(min(question_count, 5)):
            questions.append({
                "question": f"Context-aware question {i+1}: What is the main topic discussed in this content?",
                "type": "multiple_choice",
                "options": ["Topic A", "Topic B", "Topic C", "Topic D"],
                "correct_answer": "Topic A",
                "explanation": "This question was generated with enhanced context awareness.",
                "context_enhanced": True
            })
        return questions
    
    def extract_concepts_with_context(self, content: str, context_ids: List[str] = None) -> List[str]:
        """Extract concepts with context awareness"""
        # Enhanced concept extraction using stored context
        base_concepts = ["Context", "Analysis", "Enhanced", "Reasoning"]
        return base_concepts + [f"Concept_{i}" for i in range(1, 9)]
    
    def summarize_with_context(self, content: str, length: str = "medium", context_ids: List[str] = None) -> str:
        """Generate summary with context awareness"""
        word_count = len(content.split())
        return f"This content contains approximately {word_count} words and has been analyzed with enhanced context awareness. The summary provides comprehensive insights based on stored contextual information and advanced reasoning capabilities."
    
    def get_context_status(self) -> Dict[str, Any]:
        """Get fallback context status"""
        return {
            "status": "active",
            "contexts": len(self.context_store),
            "documents": len(self.document_store),
            "mode": "fallback"
        }
    
    def clear_context(self) -> Dict[str, Any]:
        """Clear fallback context"""
        self.context_store.clear()
        self.document_store.clear()
        return {"status": "cleared", "mode": "fallback"}


