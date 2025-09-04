import os
import json
import logging
import asyncio
import time
from typing import List, Dict, Optional, Any, Union, AsyncGenerator
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    """Represents a message in a conversation"""
    id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class ConversationSession:
    """Represents a conversation session"""
    session_id: str
    user_id: str
    created_at: datetime
    last_accessed: datetime
    messages: List[ConversationMessage]
    context: Dict[str, Any]
    metadata: Dict[str, Any]

class ConversationMemory:
    """
    Advanced conversation memory system with context management
    """
    
    def __init__(self, max_sessions: int = 100, max_messages_per_session: int = 1000):
        """
        Initialize conversation memory
        
        Args:
            max_sessions: Maximum number of sessions to keep
            max_messages_per_session: Maximum messages per session
        """
        self.max_sessions = max_sessions
        self.max_messages_per_session = max_messages_per_session
        self.sessions: Dict[str, ConversationSession] = {}
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)
        self.lock = threading.RLock()
        
        logger.info("Conversation memory initialized")
    
    def create_session(self, user_id: str, context: Dict[str, Any] = None) -> str:
        """
        Create a new conversation session
        
        Args:
            user_id: User identifier
            context: Initial context
            
        Returns:
            Session ID
        """
        with self.lock:
            session_id = str(uuid.uuid4())
            now = datetime.now()
            
            session = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                created_at=now,
                last_accessed=now,
                messages=[],
                context=context or {},
                metadata={}
            )
            
            self.sessions[session_id] = session
            self.user_sessions[user_id].append(session_id)
            
            # Clean up old sessions if needed
            self._cleanup_old_sessions()
            
            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id
    
    def add_message(self, session_id: str, role: str, content: str, 
                   metadata: Dict[str, Any] = None) -> str:
        """
        Add a message to a conversation session
        
        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Additional metadata
            
        Returns:
            Message ID
        """
        with self.lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            message_id = str(uuid.uuid4())
            message = ConversationMessage(
                id=message_id,
                role=role,
                content=content,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            session = self.sessions[session_id]
            session.messages.append(message)
            session.last_accessed = datetime.now()
            
            # Trim messages if needed
            if len(session.messages) > self.max_messages_per_session:
                session.messages = session.messages[-self.max_messages_per_session:]
            
            logger.debug(f"Added message {message_id} to session {session_id}")
            return message_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a conversation session"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].last_accessed = datetime.now()
                return self.sessions[session_id]
            return None
    
    def get_session_history(self, session_id: str, limit: int = 10) -> List[ConversationMessage]:
        """
        Get recent message history from a session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of recent messages
        """
        with self.lock:
            if session_id not in self.sessions:
                return []
            
            session = self.sessions[session_id]
            return session.messages[-limit:] if limit else session.messages
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all session IDs for a user"""
        with self.lock:
            return self.user_sessions.get(user_id, [])
    
    def update_session_context(self, session_id: str, context: Dict[str, Any]):
        """Update session context"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].context.update(context)
                self.sessions[session_id].last_accessed = datetime.now()
    
    def delete_session(self, session_id: str):
        """Delete a conversation session"""
        with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                user_id = session.user_id
                
                # Remove from user sessions
                if user_id in self.user_sessions:
                    self.user_sessions[user_id].remove(session_id)
                    if not self.user_sessions[user_id]:
                        del self.user_sessions[user_id]
                
                # Remove session
                del self.sessions[session_id]
                logger.info(f"Deleted session {session_id}")
    
    def _cleanup_old_sessions(self):
        """Clean up old sessions"""
        if len(self.sessions) <= self.max_sessions:
            return
        
        # Sort sessions by last accessed time
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest sessions
        sessions_to_remove = len(self.sessions) - self.max_sessions
        for session_id, _ in sorted_sessions[:sessions_to_remove]:
            self.delete_session(session_id)

class StreamingResponse:
    """
    Streaming response handler for real-time user experience
    """
    
    def __init__(self):
        self.active_streams: Dict[str, asyncio.Queue] = {}
        self.lock = threading.RLock()
    
    def create_stream(self, stream_id: str) -> asyncio.Queue:
        """Create a new streaming response"""
        with self.lock:
            queue = asyncio.Queue()
            self.active_streams[stream_id] = queue
            return queue
    
    def send_chunk(self, stream_id: str, chunk: Dict[str, Any]):
        """Send a chunk to a stream"""
        with self.lock:
            if stream_id in self.active_streams:
                try:
                    self.active_streams[stream_id].put_nowait(chunk)
                except asyncio.QueueFull:
                    logger.warning(f"Stream {stream_id} queue is full")
    
    def close_stream(self, stream_id: str):
        """Close a streaming response"""
        with self.lock:
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    async def stream_response(self, stream_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response chunks"""
        with self.lock:
            if stream_id not in self.active_streams:
                yield {"error": "Stream not found"}
                return
            
            queue = self.active_streams[stream_id]
        
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=1.0)
                    if chunk.get("type") == "end":
                        break
                    yield chunk
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield {"type": "heartbeat", "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Error in stream {stream_id}: {str(e)}")
            yield {"error": str(e)}
        finally:
            self.close_stream(stream_id)

class UserExperienceManager:
    """
    Comprehensive user experience management system
    """
    
    def __init__(self):
        self.conversation_memory = ConversationMemory()
        self.streaming_response = StreamingResponse()
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.analytics: Dict[str, Any] = defaultdict(int)
        
        logger.info("User experience manager initialized")
    
    def create_conversation(self, user_id: str, context: Dict[str, Any] = None) -> str:
        """Create a new conversation"""
        return self.conversation_memory.create_session(user_id, context)
    
    def add_user_message(self, session_id: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a user message to conversation"""
        return self.conversation_memory.add_message(session_id, "user", content, metadata)
    
    def add_assistant_message(self, session_id: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add an assistant message to conversation"""
        return self.conversation_memory.add_message(session_id, "assistant", content, metadata)
    
    def get_conversation_context(self, session_id: str, limit: int = 5) -> str:
        """
        Get conversation context for AI processing
        
        Args:
            session_id: Session identifier
            limit: Number of recent messages to include
            
        Returns:
            Formatted conversation context
        """
        messages = self.conversation_memory.get_session_history(session_id, limit)
        
        context_parts = []
        for message in messages:
            role = "User" if message.role == "user" else "Assistant"
            context_parts.append(f"{role}: {message.content}")
        
        return "\n".join(context_parts)
    
    def create_streaming_response(self, session_id: str) -> str:
        """Create a streaming response for real-time updates"""
        stream_id = str(uuid.uuid4())
        self.streaming_response.create_stream(stream_id)
        return stream_id
    
    def stream_chunk(self, stream_id: str, chunk_type: str, content: str, metadata: Dict[str, Any] = None):
        """Send a chunk to a streaming response"""
        chunk = {
            "type": chunk_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.streaming_response.send_chunk(stream_id, chunk)
    
    def end_stream(self, stream_id: str):
        """End a streaming response"""
        self.streaming_response.send_chunk(stream_id, {"type": "end"})
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        self.user_preferences[user_id].update(preferences)
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences"""
        return self.user_preferences.get(user_id, {})
    
    def track_analytics(self, event: str, metadata: Dict[str, Any] = None):
        """Track user analytics"""
        self.analytics[event] += 1
        if metadata:
            self.analytics[f"{event}_metadata"] = metadata
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics data"""
        return dict(self.analytics)
    
    async def process_with_streaming(self, session_id: str, query: str, 
                                   process_func: callable) -> str:
        """
        Process a query with streaming response
        
        Args:
            session_id: Session identifier
            query: User query
            process_func: Function to process the query
            
        Returns:
            Stream ID for the response
        """
        # Add user message
        self.add_user_message(session_id, query)
        
        # Create streaming response
        stream_id = self.create_streaming_response(session_id)
        
        # Start processing in background
        asyncio.create_task(self._process_streaming_response(
            stream_id, session_id, query, process_func
        ))
        
        return stream_id
    
    async def _process_streaming_response(self, stream_id: str, session_id: str, 
                                        query: str, process_func: callable):
        """Process streaming response in background"""
        try:
            # Send initial chunk
            self.stream_chunk(stream_id, "start", "Processing your request...")
            
            # Get conversation context
            context = self.get_conversation_context(session_id)
            
            # Process with streaming
            response_parts = []
            async for chunk in process_func(query, context):
                if chunk.get("type") == "chunk":
                    response_parts.append(chunk.get("content", ""))
                    self.stream_chunk(stream_id, "chunk", chunk.get("content", ""))
                elif chunk.get("type") == "metadata":
                    self.stream_chunk(stream_id, "metadata", "", chunk.get("data", {}))
            
            # Add assistant message
            full_response = "".join(response_parts)
            self.add_assistant_message(session_id, full_response)
            
            # End stream
            self.end_stream(stream_id)
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            self.stream_chunk(stream_id, "error", str(e))
            self.end_stream(stream_id)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of a conversation session"""
        session = self.conversation_memory.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "message_count": len(session.messages),
            "context": session.context,
            "metadata": session.metadata
        }

# Global user experience manager
user_experience_manager = UserExperienceManager()
