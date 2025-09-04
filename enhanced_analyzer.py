import os
import json
import logging
import asyncio
from typing import List, Dict, Optional, Any, AsyncGenerator
import time
from datetime import datetime

from analyzer import ContentAnalyzer
from advanced_rag import AdvancedRAG
from performance_optimizer import performance_optimizer, cached, async_cached, rate_limited
from user_experience import user_experience_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedContentAnalyzer(ContentAnalyzer):
    """
    Enhanced content analyzer with advanced RAG, performance optimization, and user experience features
    """
    
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None, 
                 enable_mcp: bool = True, enable_rag: bool = True):
        """
        Initialize enhanced analyzer
        
        Args:
            provider: AI provider ("gemini", "mistral", "huggingface")
            api_key: API key (if not provided, will use environment variable)
            enable_mcp: Enable Model Context Protocol
            enable_rag: Enable Retrieval-Augmented Generation
        """
        super().__init__(provider, api_key, enable_mcp, enable_rag)
        
        # Initialize advanced RAG system
        self.advanced_rag = AdvancedRAG(
            persist_directory="./enhanced_rag",
            use_chromadb=True,
            use_faiss=True
        )
        
        # Performance optimization
        self.performance_optimizer = performance_optimizer
        
        # User experience management
        self.user_experience = user_experience_manager
        
        logger.info("Enhanced content analyzer initialized")
    
    @async_cached(ttl=3600)
    async def enhanced_generate_recap(self, content: str, length: str = "medium", 
                                    session_id: str = None) -> str:
        """
        Enhanced recap generation with advanced RAG and streaming
        
        Args:
            content: Text content to summarize
            length: Length of summary ("brief", "medium", "detailed")
            session_id: Optional session ID for conversation context
            
        Returns:
            Generated summary
        """
        try:
            # Track analytics
            self.user_experience.track_analytics("recap_generation", {
                "length": length,
                "content_length": len(content),
                "session_id": session_id
            })
            
            # Use advanced RAG for better context
            rag_results = self.advanced_rag.hybrid_search(content, n_results=3)
            context = "\n".join([r.get("content", "") for r in rag_results])
            
            # Get conversation context if session exists
            conversation_context = ""
            if session_id:
                conversation_context = self.user_experience.get_conversation_context(session_id)
            
            # Enhanced prompt with RAG and conversation context
            context_info = f"\n\nRelevant context from similar content:\n{context}" if context else ""
            conversation_info = f"\n\nPrevious conversation:\n{conversation_context}" if conversation_context else ""
            
            prompt = f"""
            Please provide a comprehensive summary of the following content {self._get_length_instruction(length)}. 
            Focus on the main points, key concepts, and important details. 
            Make it clear and easy to understand.
            {context_info}
            {conversation_info}
            
            Content:
            {content}
            
            Summary:
            """
            
            if self.use_mock:
                return self._generate_enhanced_mock_recap(content, length, context, {})
            
            # Truncate content if too long
            max_chars = 8000
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            
            response = self._call_ai_api(prompt, max_tokens=300, temperature=0.3)
            
            # Add to conversation if session exists
            if session_id:
                self.user_experience.add_assistant_message(
                    session_id, 
                    f"Generated {length} summary", 
                    {"type": "recap", "length": length}
                )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating enhanced recap: {str(e)}")
            return self._generate_enhanced_mock_recap(content, length, "", {})
    
    @async_cached(ttl=1800)
    async def enhanced_create_questions(self, content: str, count: int = 10, 
                                      session_id: str = None) -> List[Dict]:
        """
        Enhanced question generation with advanced RAG and streaming
        
        Args:
            content: Text content to create questions from
            count: Number of questions to generate
            session_id: Optional session ID for conversation context
            
        Returns:
            List of question dictionaries
        """
        try:
            # Track analytics
            self.user_experience.track_analytics("question_generation", {
                "count": count,
                "content_length": len(content),
                "session_id": session_id
            })
            
            # Use advanced RAG for better context
            rag_results = self.advanced_rag.hybrid_search(content, n_results=5)
            context = "\n".join([r.get("content", "") for r in rag_results])
            
            # Get conversation context if session exists
            conversation_context = ""
            if session_id:
                conversation_context = self.user_experience.get_conversation_context(session_id)
            
            # Enhanced prompt with RAG and conversation context
            context_info = f"\n\nRelevant context from similar content:\n{context}" if context else ""
            conversation_info = f"\n\nPrevious conversation:\n{conversation_context}" if conversation_context else ""
            
            prompt = f"""
            Create {count} quiz questions based on the following content. 
            Include a mix of question types:
            - Multiple choice (4 options each)
            - True/False
            - Short answer
            
            For each question, provide:
            1. The question text
            2. Question type (multiple_choice, true_false, short_answer)
            3. Options (for multiple choice)
            4. Correct answer
            5. Explanation (brief)
            {context_info}
            {conversation_info}
            
            Format as JSON array with this structure:
            [
                {{
                    "question": "Question text here",
                    "type": "multiple_choice",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "A",
                    "explanation": "Brief explanation"
                }}
            ]
            
            Content:
            {content}
            """
            
            if self.use_mock:
                return self._generate_enhanced_mock_questions(content, count, context, [])
            
            # Truncate content if too long
            max_chars = 6000
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            
            response = self._call_ai_api(prompt, max_tokens=1500, temperature=0.5)
            
            # Clean up the response
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            questions = json.loads(response)
            validated_questions = self._validate_questions(questions)
            
            # Add to conversation if session exists
            if session_id:
                self.user_experience.add_assistant_message(
                    session_id, 
                    f"Generated {len(validated_questions)} quiz questions", 
                    {"type": "questions", "count": len(validated_questions)}
                )
            
            return validated_questions
            
        except Exception as e:
            logger.error(f"Error creating enhanced questions: {str(e)}")
            return self._generate_enhanced_mock_questions(content, count, "", [])
    
    @async_cached(ttl=7200)
    async def enhanced_extract_key_concepts(self, content: str, 
                                          session_id: str = None) -> List[str]:
        """
        Enhanced key concept extraction with advanced RAG
        
        Args:
            content: Text content to analyze
            session_id: Optional session ID for conversation context
            
        Returns:
            List of key concepts
        """
        try:
            # Track analytics
            self.user_experience.track_analytics("concept_extraction", {
                "content_length": len(content),
                "session_id": session_id
            })
            
            # Use advanced RAG for better context
            rag_results = self.advanced_rag.hybrid_search(content, n_results=3)
            context = "\n".join([r.get("content", "") for r in rag_results])
            
            # Enhanced prompt with RAG context
            context_info = f"\n\nRelevant context from similar content:\n{context}" if context else ""
            
            prompt = f"""
            Extract the 8-12 most important key concepts, topics, or themes from the following content.
            Return them as a simple list, one concept per line.
            Focus on the main ideas, important terms, and central topics.
            {context_info}
            
            Content:
            {content}
            
            Key concepts:
            """
            
            if self.use_mock:
                return self._extract_mock_concepts(content)
            
            # Truncate content if too long
            max_chars = 6000
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            
            response = self._call_ai_api(prompt, max_tokens=200, temperature=0.3)
            
            # Parse concepts
            concepts = []
            for line in response.split('\n'):
                concept = line.strip()
                if concept and not concept.startswith('-') and not concept.startswith('•'):
                    # Remove numbering if present
                    import re
                    concept = re.sub(r'^\d+\.\s*', '', concept)
                    concepts.append(concept)
            
            # Add to conversation if session exists
            if session_id:
                self.user_experience.add_assistant_message(
                    session_id, 
                    f"Extracted {len(concepts)} key concepts", 
                    {"type": "concepts", "count": len(concepts)}
                )
            
            return concepts[:12]  # Limit to 12 concepts
            
        except Exception as e:
            logger.error(f"Error extracting enhanced key concepts: {str(e)}")
            return self._extract_mock_concepts(content)
    
    async def enhanced_ask_learning_question(self, question: str, content_id: str = None, 
                                           context: str = None, session_id: str = None) -> Dict[str, Any]:
        """
        Enhanced learning Q&A with advanced RAG, streaming, and conversation context
        
        Args:
            question: The user's learning question
            content_id: Optional content ID to retrieve from RAG
            context: Optional additional context
            session_id: Optional session ID for conversation context
            
        Returns:
            Dictionary with answer, sources, and learning insights
        """
        try:
            # Track analytics
            self.user_experience.track_analytics("learning_question", {
                "question_length": len(question),
                "has_content_id": bool(content_id),
                "has_context": bool(context),
                "session_id": session_id
            })
            
            # Add user question to conversation if session exists
            if session_id:
                self.user_experience.add_user_message(session_id, question)
            
            # Get relevant context from advanced RAG
            rag_results = []
            if content_id:
                rag_results = self.advanced_rag.search_similar_content(f"content_id:{content_id}", n_results=3)
            else:
                rag_results = self.advanced_rag.hybrid_search(question, n_results=5)
            
            # Get conversation context
            conversation_context = ""
            if session_id:
                conversation_context = self.user_experience.get_conversation_context(session_id)
            
            # Build comprehensive prompt
            rag_context = "\n\n".join([r.get("content", "") for r in rag_results[:3]])
            
            prompt = f"""
            You are an expert learning assistant. Answer the following learning question using the provided context, 
            your knowledge, and conversation history.
            
            LEARNING QUESTION: {question}
            
            RELEVANT CONTEXT FROM DOCUMENTS:
            {rag_context[:2000]}
            
            CONVERSATION HISTORY:
            {conversation_context[:1000]}
            
            ADDITIONAL CONTEXT:
            {context or "None provided"}
            
            INSTRUCTIONS:
            1. Provide a comprehensive, educational answer focused on learning
            2. Structure your response with clear explanations and examples
            3. Include learning insights and connections to broader concepts
            4. Reference the conversation history when relevant
            5. If the question is not learning-related, politely redirect to educational topics
            
            Please provide a detailed, educational response that helps the user learn and understand the topic better.
            """
            
            if self.use_mock:
                return self._generate_mock_learning_answer(question, rag_results)
            
            # Use AI API for comprehensive answers
            response = self._call_ai_api(prompt, max_tokens=2048, temperature=0.7)
            
            # Extract sources and learning insights
            sources = [{"type": "document", "content": r.get("content", "")[:200]} for r in rag_results[:3]]
            if conversation_context:
                sources.append({"type": "conversation", "content": conversation_context[:200]})
            
            result = {
                "answer": response,
                "sources": sources,
                "learning_insights": self._extract_learning_insights(response),
                "question_type": self._classify_question_type(question),
                "confidence": self._calculate_confidence(rag_results, {}),
                "rag_context_used": len(rag_results) > 0,
                "conversation_context_used": bool(conversation_context)
            }
            
            # Add assistant response to conversation if session exists
            if session_id:
                self.user_experience.add_assistant_message(
                    session_id, 
                    response, 
                    {"type": "learning_answer", "confidence": result["confidence"]}
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error answering enhanced learning question: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "learning_insights": [],
                "question_type": "error",
                "confidence": 0.0,
                "rag_context_used": False,
                "conversation_context_used": False
            }
    
    async def stream_analysis(self, content: str, analysis_type: str, 
                            session_id: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream analysis results for real-time user experience
        
        Args:
            content: Content to analyze
            analysis_type: Type of analysis ("recap", "questions", "concepts")
            session_id: Optional session ID
            
        Yields:
            Analysis chunks
        """
        try:
            # Create streaming response
            stream_id = self.user_experience.create_streaming_response(session_id or "default")
            
            # Send initial chunk
            yield {"type": "start", "analysis_type": analysis_type, "stream_id": stream_id}
            
            # Process based on analysis type
            if analysis_type == "recap":
                result = await self.enhanced_generate_recap(content, session_id=session_id)
                yield {"type": "chunk", "content": result}
            elif analysis_type == "questions":
                questions = await self.enhanced_create_questions(content, session_id=session_id)
                for i, question in enumerate(questions):
                    yield {"type": "chunk", "content": f"Question {i+1}: {question.get('question', '')}"}
            elif analysis_type == "concepts":
                concepts = await self.enhanced_extract_key_concepts(content, session_id=session_id)
                for concept in concepts:
                    yield {"type": "chunk", "content": f"• {concept}"}
            
            # Send completion chunk
            yield {"type": "complete", "analysis_type": analysis_type}
            
        except Exception as e:
            logger.error(f"Error in streaming analysis: {str(e)}")
            yield {"type": "error", "error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "analyzer_metrics": {
                "provider": self.provider,
                "use_mock": self.use_mock,
                "enable_mcp": self.enable_mcp,
                "enable_rag": self.enable_rag
            },
            "rag_metrics": self.advanced_rag.get_performance_stats(),
            "performance_metrics": self.performance_optimizer.get_metrics(),
            "user_experience_metrics": self.user_experience.get_analytics()
        }
    
    def _get_length_instruction(self, length: str) -> str:
        """Get length instruction for prompts"""
        length_instructions = {
            "brief": "in 2-3 sentences",
            "medium": "in 4-6 sentences", 
            "detailed": "in 8-12 sentences"
        }
        return length_instructions.get(length, length_instructions["medium"])
    
    def create_conversation_session(self, user_id: str, context: Dict[str, Any] = None) -> str:
        """Create a new conversation session"""
        return self.user_experience.create_conversation(user_id, context)
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history"""
        messages = self.user_experience.conversation_memory.get_session_history(session_id, limit)
        return [{"role": msg.role, "content": msg.content, "timestamp": msg.timestamp.isoformat()} for msg in messages]
    
    def clear_performance_cache(self):
        """Clear all performance caches"""
        self.performance_optimizer.clear_cache()
        self.advanced_rag.clear_cache()
        logger.info("All performance caches cleared")
