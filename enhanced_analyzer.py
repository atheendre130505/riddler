import os
import json
import logging
import asyncio
from typing import List, Dict, Optional, Any, AsyncGenerator
import time
from datetime import datetime

from advanced_rag import AdvancedRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """Base content analyzer class"""
    
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None, 
                 enable_mcp: bool = True, enable_rag: bool = True):
        self.provider = provider
        self.api_key = api_key
        self.enable_mcp = enable_mcp
        self.enable_rag = enable_rag
        
        # Initialize AI provider
        if provider == "gemini":
            try:
                import google.generativeai as genai
                if api_key:
                    genai.configure(api_key=api_key)
                else:
                    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.model = genai.GenerativeModel('gemini-pro')
                logger.info("Gemini AI initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.model = None
        else:
            self.model = None

class EnhancedContentAnalyzer(ContentAnalyzer):
    """
    Enhanced content analyzer with advanced RAG
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
        
        logger.info("Enhanced content analyzer initialized")
    
    async def enhanced_generate_recap(self, content: str, length: str = "medium", 
                                    session_id: str = None) -> str:
        """
        Enhanced recap generation with advanced RAG
        
        Args:
            content: Text content to summarize
            length: Length of summary ("brief", "medium", "detailed")
            session_id: Optional session ID for conversation context
            
        Returns:
            Generated summary
        """
        try:
            if not self.model:
                return self._generate_mock_recap(content, length)
            
            # Add content to RAG system for context
            if self.enable_rag and self.advanced_rag:
                self.advanced_rag.add_document(content, metadata={"type": "content", "session_id": session_id})
            
            # Generate summary using AI
            prompt = f"""
            Please provide a {length} summary of the following content:
            
            {content}
            
            The summary should be:
            - Clear and concise
            - Capture the main points
            - Be appropriate for learning purposes
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.model.generate_content, prompt
            )
            
            return response.text if response and response.text else self._generate_mock_recap(content, length)
            
        except Exception as e:
            logger.error(f"Error generating enhanced recap: {str(e)}")
            return self._generate_mock_recap(content, length)
    
    async def enhanced_create_questions(self, content: str, count: int = 10, 
                                      session_id: str = None) -> List[Dict]:
        """
        Enhanced question generation with advanced RAG
        
        Args:
            content: Text content to generate questions from
            count: Number of questions to generate
            session_id: Optional session ID for conversation context
            
        Returns:
            List of generated questions
        """
        try:
            if not self.model:
                return self._generate_mock_questions(content, count)
            
            # Add content to RAG system for context
            if self.enable_rag and self.advanced_rag:
                self.advanced_rag.add_document(content, metadata={"type": "content", "session_id": session_id})
            
            # Generate questions using AI
            prompt = f"""
            Generate {count} diverse questions based on the following content:
            
            {content}
            
            Create questions that:
            - Test understanding of key concepts
            - Include multiple choice, true/false, and short answer questions
            - Are appropriate for learning and assessment
            - Cover different difficulty levels
            
            Return the questions in JSON format with the following structure:
            {{
                "questions": [
                    {{
                        "question": "Question text",
                        "type": "multiple_choice|true_false|short_answer|fill_in_blank",
                        "options": ["option1", "option2", "option3", "option4"] (for multiple choice),
                        "correct_answer": "correct answer",
                        "explanation": "explanation of the answer",
                        "difficulty": "easy|medium|hard",
                        "points": 2
                    }}
                ]
            }}
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.model.generate_content, prompt
            )
            
            if response and response.text:
                try:
                    # Try to parse JSON response
                    result = json.loads(response.text)
                    return result.get("questions", self._generate_mock_questions(content, count))
                except json.JSONDecodeError:
                    # If JSON parsing fails, return mock questions
                    return self._generate_mock_questions(content, count)
            else:
                return self._generate_mock_questions(content, count)
            
        except Exception as e:
            logger.error(f"Error creating enhanced questions: {str(e)}")
            return self._generate_mock_questions(content, count)
    
    async def enhanced_extract_key_concepts(self, content: str, 
                                          session_id: str = None) -> List[str]:
        """
        Enhanced key concept extraction with advanced RAG
        
        Args:
            content: Text content to extract concepts from
            session_id: Optional session ID for conversation context
            
        Returns:
            List of key concepts
        """
        try:
            if not self.model:
                return self._extract_mock_concepts(content)
            
            # Add content to RAG system for context
            if self.enable_rag and self.advanced_rag:
                self.advanced_rag.add_document(content, metadata={"type": "content", "session_id": session_id})
            
            # Extract concepts using AI
            prompt = f"""
            Extract the key concepts from the following content:
            
            {content}
            
            Return a list of the most important concepts, terms, or topics.
            Focus on concepts that are central to understanding the content.
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.model.generate_content, prompt
            )
            
            if response and response.text:
                # Parse the response to extract concepts
                concepts = []
                lines = response.text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('*'):
                        # Remove numbering and bullet points
                        concept = line.lstrip('0123456789.-* ').strip()
                        if concept:
                            concepts.append(concept)
                return concepts[:10]  # Limit to 10 concepts
            else:
                return self._extract_mock_concepts(content)
            
        except Exception as e:
            logger.error(f"Error extracting key concepts: {str(e)}")
            return self._extract_mock_concepts(content)
    
    async def enhanced_ask_learning_question(self, question: str, context: str = "", 
                                           session_id: str = None) -> str:
        """
        Enhanced learning question answering with advanced RAG
        
        Args:
            question: User's question
            context: Additional context
            session_id: Optional session ID for conversation context
            
        Returns:
            AI-generated answer
        """
        try:
            if not self.model:
                return self._generate_mock_answer(question)
            
            # Search for relevant context using RAG
            relevant_context = ""
            if self.enable_rag and self.advanced_rag and context:
                try:
                    search_results = self.advanced_rag.hybrid_search(question, k=3)
                    if search_results:
                        relevant_context = "\n".join([result.get("content", "") for result in search_results])
                except Exception as e:
                    logger.warning(f"RAG search failed: {e}")
            
            # Generate answer using AI
            prompt = f"""
            Answer the following question in a helpful, educational way:
            
            Question: {question}
            
            Context: {context}
            Relevant Information: {relevant_context}
            
            Please provide a clear, informative answer that helps with learning.
            If you don't have enough information, say so and suggest where to find more details.
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.model.generate_content, prompt
            )
            
            return response.text if response and response.text else self._generate_mock_answer(question)
            
        except Exception as e:
            logger.error(f"Error answering learning question: {str(e)}")
            return self._generate_mock_answer(question)
    
    def _generate_mock_recap(self, content: str, length: str) -> str:
        """Generate a mock recap when AI is not available"""
        words = content.split()
        if length == "brief":
            summary_length = min(50, len(words))
        elif length == "detailed":
            summary_length = min(200, len(words))
        else:  # medium
            summary_length = min(100, len(words))
        
        return " ".join(words[:summary_length]) + "..." if len(words) > summary_length else content
    
    def _generate_mock_questions(self, content: str, count: int) -> List[Dict]:
        """Generate mock questions when AI is not available"""
        questions = []
        words = content.split()
        
        for i in range(min(count, 5)):  # Limit to 5 mock questions
            questions.append({
                "question": f"Question {i+1}: What is the main topic discussed in this content?",
                "type": "multiple_choice",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "Option A",
                "explanation": "This is a mock question for demonstration purposes.",
                "difficulty": "medium",
                "points": 2
            })
        
        return questions
    
    def _extract_mock_concepts(self, content: str) -> List[str]:
        """Extract mock concepts when AI is not available"""
        words = content.split()
        # Simple concept extraction based on word frequency
        word_freq = {}
        for word in words:
            word = word.lower().strip('.,!?;:"()[]{}')
            if len(word) > 3:  # Only consider words longer than 3 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top 5 most frequent words as concepts
        concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [concept[0] for concept in concepts]
    
    def _generate_mock_answer(self, question: str) -> str:
        """Generate a mock answer when AI is not available"""
        return f"This is a mock answer to: {question}. The AI service is not available, but this demonstrates the functionality."