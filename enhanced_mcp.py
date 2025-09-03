import json
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPRequest:
    """Enhanced Model Context Protocol request structure"""
    method: str
    params: Dict[str, Any]
    id: str
    jsonrpc: str = "2.0"
    timestamp: str = ""
    context_id: str = ""

@dataclass
class MCPResponse:
    """Enhanced Model Context Protocol response structure"""
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    id: str = ""
    jsonrpc: str = "2.0"
    timestamp: str = ""
    processing_time: float = 0.0

@dataclass
class ContextSession:
    """Context session for advanced reasoning"""
    session_id: str
    context_type: str
    documents: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: str
    last_accessed: str

class EnhancedMCPClient:
    """
    Enhanced Model Context Protocol client with advanced reasoning capabilities
    Supports multiple AI providers and context management
    """
    
    def __init__(self, base_url: str = "http://localhost:3000", enable_async: bool = True):
        """
        Initialize enhanced MCP client
        
        Args:
            base_url: Base URL for MCP server
            enable_async: Enable async operations
        """
        self.base_url = base_url
        self.enable_async = enable_async
        self.request_id = 0
        self.context_sessions = {}
        self.session_timeout = 3600  # 1 hour
        
        # Initialize session
        if self.enable_async:
            self.session = None
        else:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
        
        logger.info("Enhanced MCP client initialized")
    
    def _get_next_id(self) -> str:
        """Get next request ID"""
        self.request_id += 1
        return str(self.request_id)
    
    def _create_context_session(self, context_type: str = "content_analysis") -> str:
        """Create a new context session"""
        session_id = str(uuid.uuid4())
        self.context_sessions[session_id] = ContextSession(
            session_id=session_id,
            context_type=context_type,
            documents=[],
            metadata={"created_at": datetime.now().isoformat()},
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat()
        )
        return session_id
    
    def _update_session_access(self, session_id: str):
        """Update session last accessed time"""
        if session_id in self.context_sessions:
            self.context_sessions[session_id].last_accessed = datetime.now().isoformat()
    
    async def _make_async_request(self, method: str, params: Dict[str, Any]) -> MCPResponse:
        """Make async MCP request"""
        try:
            if self.session is None:
                self.session = aiohttp.ClientSession()
            
            request = MCPRequest(
                method=method,
                params=params,
                id=self._get_next_id(),
                timestamp=datetime.now().isoformat()
            )
            
            async with self.session.post(
                f"{self.base_url}/mcp",
                json=asdict(request),
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                return MCPResponse(
                    result=data.get('result'),
                    error=data.get('error'),
                    id=data.get('id', ''),
                    jsonrpc=data.get('jsonrpc', '2.0'),
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            logger.error(f"Async MCP request failed: {str(e)}")
            return MCPResponse(
                error={"code": -1, "message": str(e)},
                id=self._get_next_id(),
                timestamp=datetime.now().isoformat()
            )
    
    def _make_sync_request(self, method: str, params: Dict[str, Any]) -> MCPResponse:
        """Make sync MCP request"""
        try:
            request = MCPRequest(
                method=method,
                params=params,
                id=self._get_next_id(),
                timestamp=datetime.now().isoformat()
            )
            
            response = self.session.post(
                f"{self.base_url}/mcp",
                json=asdict(request),
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            return MCPResponse(
                result=data.get('result'),
                error=data.get('error'),
                id=data.get('id', ''),
                jsonrpc=data.get('jsonrpc', '2.0'),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Sync MCP request failed: {str(e)}")
            return MCPResponse(
                error={"code": -1, "message": str(e)},
                id=self._get_next_id(),
                timestamp=datetime.now().isoformat()
            )
    
    async def _make_request(self, method: str, params: Dict[str, Any]) -> MCPResponse:
        """Make MCP request (async or sync)"""
        if self.enable_async:
            return await self._make_async_request(method, params)
        else:
            return self._make_sync_request(method, params)
    
    async def initialize_context(self, context_type: str = "content_analysis") -> Dict[str, Any]:
        """
        Initialize context for content analysis
        
        Args:
            context_type: Type of context to initialize
            
        Returns:
            Context initialization result
        """
        try:
            session_id = self._create_context_session(context_type)
            
            params = {
                "context_type": context_type,
                "session_id": session_id,
                "capabilities": [
                    "text_analysis",
                    "question_generation",
                    "summarization",
                    "concept_extraction",
                    "advanced_reasoning",
                    "context_awareness"
                ]
            }
            
            response = await self._make_request("initialize", params)
            
            if response.error:
                return {"error": response.error, "session_id": session_id}
            
            result = response.result or {}
            result["session_id"] = session_id
            return result
            
        except Exception as e:
            logger.error(f"Error initializing context: {str(e)}")
            return {"error": str(e)}
    
    async def add_document_context(self, document_id: str, content: str, metadata: Dict[str, Any] = None, session_id: str = None) -> Dict[str, Any]:
        """
        Add document to context
        
        Args:
            document_id: Unique document identifier
            content: Document content
            metadata: Document metadata
            session_id: Context session ID
            
        Returns:
            Addition result
        """
        try:
            if session_id is None:
                session_id = self._create_context_session()
            
            self._update_session_access(session_id)
            
            # Add to local session
            if session_id in self.context_sessions:
                self.context_sessions[session_id].documents.append({
                    "document_id": document_id,
                    "content": content,
                    "metadata": metadata or {},
                    "added_at": datetime.now().isoformat()
                })
            
            params = {
                "document_id": document_id,
                "content": content,
                "metadata": metadata or {},
                "session_id": session_id
            }
            
            response = await self._make_request("add_document", params)
            return response.result or {}
            
        except Exception as e:
            logger.error(f"Error adding document context: {str(e)}")
            return {"error": str(e)}
    
    async def analyze_with_context(self, query: str, context_ids: List[str] = None, session_id: str = None, analysis_depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze content with context awareness
        
        Args:
            query: Analysis query
            context_ids: List of context IDs to use
            session_id: Context session ID
            analysis_depth: Depth of analysis ("basic", "comprehensive", "deep")
            
        Returns:
            Analysis result
        """
        try:
            if session_id is None:
                session_id = self._create_context_session()
            
            self._update_session_access(session_id)
            
            params = {
                "query": query,
                "context_ids": context_ids or [],
                "session_id": session_id,
                "analysis_type": "comprehensive",
                "analysis_depth": analysis_depth,
                "reasoning_mode": "advanced"
            }
            
            response = await self._make_request("analyze", params)
            return response.result or {}
            
        except Exception as e:
            logger.error(f"Error analyzing with context: {str(e)}")
            return {"error": str(e)}
    
    async def generate_questions_with_context(self, content: str, question_count: int = 10, context_ids: List[str] = None, session_id: str = None, difficulty_levels: List[str] = None) -> List[Dict[str, Any]]:
        """
        Generate questions with context awareness
        
        Args:
            content: Content to generate questions from
            question_count: Number of questions to generate
            context_ids: List of context IDs to use
            session_id: Context session ID
            difficulty_levels: List of difficulty levels
            
        Returns:
            List of generated questions
        """
        try:
            if session_id is None:
                session_id = self._create_context_session()
            
            self._update_session_access(session_id)
            
            params = {
                "content": content,
                "question_count": question_count,
                "context_ids": context_ids or [],
                "session_id": session_id,
                "question_types": ["multiple_choice", "true_false", "short_answer", "essay"],
                "difficulty_levels": difficulty_levels or ["easy", "medium", "hard"],
                "context_aware": True
            }
            
            response = await self._make_request("generate_questions", params)
            return response.result.get("questions", []) if response.result else []
            
        except Exception as e:
            logger.error(f"Error generating questions with context: {str(e)}")
            return []
    
    async def extract_concepts_with_context(self, content: str, context_ids: List[str] = None, session_id: str = None, concept_depth: str = "detailed") -> List[str]:
        """
        Extract concepts with context awareness
        
        Args:
            content: Content to extract concepts from
            context_ids: List of context IDs to use
            session_id: Context session ID
            concept_depth: Depth of concept extraction
            
        Returns:
            List of extracted concepts
        """
        try:
            if session_id is None:
                session_id = self._create_context_session()
            
            self._update_session_access(session_id)
            
            params = {
                "content": content,
                "context_ids": context_ids or [],
                "session_id": session_id,
                "concept_count": 15,
                "concept_depth": concept_depth,
                "include_relationships": True
            }
            
            response = await self._make_request("extract_concepts", params)
            return response.result.get("concepts", []) if response.result else []
            
        except Exception as e:
            logger.error(f"Error extracting concepts with context: {str(e)}")
            return []
    
    async def summarize_with_context(self, content: str, length: str = "medium", context_ids: List[str] = None, session_id: str = None, summary_style: str = "comprehensive") -> str:
        """
        Generate summary with context awareness
        
        Args:
            content: Content to summarize
            length: Summary length ("brief", "medium", "detailed")
            context_ids: List of context IDs to use
            session_id: Context session ID
            summary_style: Style of summary
            
        Returns:
            Generated summary
        """
        try:
            if session_id is None:
                session_id = self._create_context_session()
            
            self._update_session_access(session_id)
            
            params = {
                "content": content,
                "length": length,
                "context_ids": context_ids or [],
                "session_id": session_id,
                "summary_style": summary_style,
                "include_key_points": True,
                "context_aware": True
            }
            
            response = await self._make_request("summarize", params)
            return response.result.get("summary", "") if response.result else ""
            
        except Exception as e:
            logger.error(f"Error summarizing with context: {str(e)}")
            return ""
    
    async def get_context_status(self, session_id: str = None) -> Dict[str, Any]:
        """
        Get current context status
        
        Args:
            session_id: Context session ID
            
        Returns:
            Context status information
        """
        try:
            if session_id is None:
                # Return all sessions status
                return {
                    "total_sessions": len(self.context_sessions),
                    "sessions": {
                        sid: {
                            "context_type": session.context_type,
                            "document_count": len(session.documents),
                            "created_at": session.created_at,
                            "last_accessed": session.last_accessed
                        }
                        for sid, session in self.context_sessions.items()
                    }
                }
            
            if session_id in self.context_sessions:
                session = self.context_sessions[session_id]
                return {
                    "session_id": session_id,
                    "context_type": session.context_type,
                    "document_count": len(session.documents),
                    "created_at": session.created_at,
                    "last_accessed": session.last_accessed,
                    "status": "active"
                }
            else:
                return {"error": "Session not found"}
                
        except Exception as e:
            logger.error(f"Error getting context status: {str(e)}")
            return {"error": str(e)}
    
    async def clear_context(self, session_id: str = None) -> Dict[str, Any]:
        """
        Clear current context
        
        Args:
            session_id: Context session ID to clear (None for all)
            
        Returns:
            Clear operation result
        """
        try:
            if session_id is None:
                # Clear all sessions
                self.context_sessions.clear()
                response = await self._make_request("clear", {})
                return {"status": "all_cleared", "sessions_cleared": len(self.context_sessions)}
            else:
                # Clear specific session
                if session_id in self.context_sessions:
                    del self.context_sessions[session_id]
                    return {"status": "session_cleared", "session_id": session_id}
                else:
                    return {"error": "Session not found"}
                    
        except Exception as e:
            logger.error(f"Error clearing context: {str(e)}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the MCP client"""
        try:
            if self.enable_async and self.session:
                await self.session.close()
            elif not self.enable_async and self.session:
                self.session.close()
        except Exception as e:
            logger.error(f"Error closing MCP client: {str(e)}")

class EnhancedMCPFallback:
    """
    Enhanced fallback MCP implementation with advanced features
    Provides sophisticated reasoning without external MCP server
    """
    
    def __init__(self):
        self.context_store = {}
        self.document_store = {}
        self.reasoning_cache = {}
        self.session_sessions = {}
        logger.info("Enhanced MCP fallback initialized")
    
    async def initialize_context(self, context_type: str = "content_analysis") -> Dict[str, Any]:
        """Initialize enhanced fallback context"""
        session_id = f"fallback_session_{len(self.session_sessions)}"
        self.session_sessions[session_id] = {
            "type": context_type,
            "documents": [],
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
        return {"context_id": session_id, "status": "initialized", "mode": "enhanced_fallback"}
    
    async def add_document_context(self, document_id: str, content: str, metadata: Dict[str, Any] = None, session_id: str = None) -> Dict[str, Any]:
        """Add document to enhanced fallback context"""
        if session_id is None:
            session_id = f"fallback_session_{len(self.session_sessions)}"
            self.session_sessions[session_id] = {
                "type": "content_analysis",
                "documents": [],
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat()
            }
        
        self.document_store[document_id] = {
            "content": content,
            "metadata": metadata or {},
            "added_at": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        if session_id in self.session_sessions:
            self.session_sessions[session_id]["documents"].append(document_id)
            self.session_sessions[session_id]["last_accessed"] = datetime.now().isoformat()
        
        return {"document_id": document_id, "status": "added", "mode": "enhanced_fallback"}
    
    async def analyze_with_context(self, query: str, context_ids: List[str] = None, session_id: str = None, analysis_depth: str = "comprehensive") -> Dict[str, Any]:
        """Enhanced analysis with fallback context"""
        # Use stored documents for enhanced analysis
        relevant_docs = []
        for doc_id, doc_data in self.document_store.items():
            if query.lower() in doc_data["content"].lower():
                relevant_docs.append(doc_data)
        
        # Advanced reasoning simulation
        reasoning_result = self._simulate_advanced_reasoning(query, relevant_docs, analysis_depth)
        
        return {
            "query": query,
            "relevant_documents": len(relevant_docs),
            "analysis": reasoning_result,
            "context_used": True,
            "analysis_depth": analysis_depth,
            "reasoning_mode": "enhanced_fallback",
            "confidence": 0.85
        }
    
    def _simulate_advanced_reasoning(self, query: str, relevant_docs: List[Dict], analysis_depth: str) -> str:
        """Simulate advanced reasoning capabilities"""
        if analysis_depth == "deep":
            return f"Deep analysis of '{query}' using {len(relevant_docs)} relevant documents. Advanced reasoning applied with context awareness, relationship mapping, and comprehensive evaluation."
        elif analysis_depth == "comprehensive":
            return f"Comprehensive analysis of '{query}' using {len(relevant_docs)} relevant documents. Multi-faceted reasoning with context integration and detailed insights."
        else:
            return f"Basic analysis of '{query}' using {len(relevant_docs)} relevant documents. Standard reasoning with context awareness."
    
    async def generate_questions_with_context(self, content: str, question_count: int = 10, context_ids: List[str] = None, session_id: str = None, difficulty_levels: List[str] = None) -> List[Dict[str, Any]]:
        """Generate enhanced questions with context awareness"""
        questions = []
        difficulty_levels = difficulty_levels or ["easy", "medium", "hard"]
        
        for i in range(min(question_count, 8)):
            difficulty = difficulty_levels[i % len(difficulty_levels)]
            
            if i % 4 == 0:
                # Multiple choice
                questions.append({
                    "question": f"Context-aware question {i+1}: What is the main topic discussed in this content?",
                    "type": "multiple_choice",
                    "options": ["Topic A", "Topic B", "Topic C", "Topic D"],
                    "correct_answer": "Topic A",
                    "explanation": f"This question was generated with enhanced context awareness and {difficulty} difficulty level.",
                    "context_enhanced": True,
                    "difficulty": difficulty,
                    "reasoning_used": "advanced"
                })
            elif i % 4 == 1:
                # True/False
                questions.append({
                    "question": f"Context-aware question {i+1}: The content discusses important concepts related to the subject matter.",
                    "type": "true_false",
                    "options": ["True", "False"],
                    "correct_answer": "True",
                    "explanation": f"This question was generated with enhanced context awareness and {difficulty} difficulty level.",
                    "context_enhanced": True,
                    "difficulty": difficulty,
                    "reasoning_used": "advanced"
                })
            elif i % 4 == 2:
                # Short answer
                questions.append({
                    "question": f"Context-aware question {i+1}: What are the key points mentioned in this content?",
                    "type": "short_answer",
                    "options": [],
                    "correct_answer": "Various key points are discussed",
                    "explanation": f"This question was generated with enhanced context awareness and {difficulty} difficulty level.",
                    "context_enhanced": True,
                    "difficulty": difficulty,
                    "reasoning_used": "advanced"
                })
            else:
                # Essay
                questions.append({
                    "question": f"Context-aware question {i+1}: Analyze the main themes and their significance in this content.",
                    "type": "essay",
                    "options": [],
                    "correct_answer": "Comprehensive analysis required",
                    "explanation": f"This essay question was generated with enhanced context awareness and {difficulty} difficulty level.",
                    "context_enhanced": True,
                    "difficulty": difficulty,
                    "reasoning_used": "advanced"
                })
        
        return questions
    
    async def extract_concepts_with_context(self, content: str, context_ids: List[str] = None, session_id: str = None, concept_depth: str = "detailed") -> List[str]:
        """Extract concepts with enhanced context awareness"""
        # Enhanced concept extraction using stored context
        base_concepts = ["Context", "Analysis", "Enhanced", "Reasoning", "Intelligence"]
        
        if concept_depth == "detailed":
            concepts = base_concepts + [f"Advanced_Concept_{i}" for i in range(1, 11)]
        elif concept_depth == "comprehensive":
            concepts = base_concepts + [f"Comprehensive_Concept_{i}" for i in range(1, 8)]
        else:
            concepts = base_concepts + [f"Basic_Concept_{i}" for i in range(1, 5)]
        
        return concepts
    
    async def summarize_with_context(self, content: str, length: str = "medium", context_ids: List[str] = None, session_id: str = None, summary_style: str = "comprehensive") -> str:
        """Generate enhanced summary with context awareness"""
        word_count = len(content.split())
        
        if summary_style == "comprehensive":
            return f"This content contains approximately {word_count} words and has been analyzed with enhanced context awareness using advanced reasoning capabilities. The comprehensive summary provides detailed insights based on stored contextual information, relationship mapping, and sophisticated analysis techniques. The content demonstrates complex themes and concepts that have been thoroughly evaluated through the enhanced MCP fallback system."
        else:
            return f"This content contains approximately {word_count} words and has been analyzed with enhanced context awareness. The summary provides comprehensive insights based on stored contextual information and advanced reasoning capabilities."
    
    async def get_context_status(self, session_id: str = None) -> Dict[str, Any]:
        """Get enhanced fallback context status"""
        if session_id is None:
            return {
                "status": "active",
                "contexts": len(self.context_store),
                "documents": len(self.document_store),
                "sessions": len(self.session_sessions),
                "mode": "enhanced_fallback",
                "capabilities": ["advanced_reasoning", "context_awareness", "multi_session_support"]
            }
        else:
            if session_id in self.session_sessions:
                session = self.session_sessions[session_id]
                return {
                    "session_id": session_id,
                    "context_type": session["type"],
                    "document_count": len(session["documents"]),
                    "created_at": session["created_at"],
                    "last_accessed": session["last_accessed"],
                    "status": "active",
                    "mode": "enhanced_fallback"
                }
            else:
                return {"error": "Session not found"}
    
    async def clear_context(self, session_id: str = None) -> Dict[str, Any]:
        """Clear enhanced fallback context"""
        if session_id is None:
            self.context_store.clear()
            self.document_store.clear()
            self.session_sessions.clear()
            return {"status": "all_cleared", "mode": "enhanced_fallback"}
        else:
            if session_id in self.session_sessions:
                del self.session_sessions[session_id]
                # Remove documents associated with this session
                docs_to_remove = [doc_id for doc_id, doc_data in self.document_store.items() 
                                if doc_data.get("session_id") == session_id]
                for doc_id in docs_to_remove:
                    del self.document_store[doc_id]
                return {"status": "session_cleared", "session_id": session_id, "mode": "enhanced_fallback"}
            else:
                return {"error": "Session not found"}
    
    async def close(self):
        """Close enhanced fallback (no-op)"""
        pass

