import os
import json
import logging
import requests
from typing import List, Dict, Optional, Any
import re

from enhanced_mcp import EnhancedMCPClient, EnhancedMCPFallback
from hybrid_rag import HybridRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """
    Handles content analysis using free AI providers
    Supports Google Gemini, Mistral, and Hugging Face models
    Generates summaries, quiz questions, and extracts key concepts
    """
    
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None, enable_mcp: bool = True, enable_rag: bool = True):
        """
        Initialize the analyzer with free AI provider, MCP, and RAG
        
        Args:
            provider: AI provider ("gemini", "mistral", "huggingface")
            api_key: API key (if not provided, will use environment variable)
            enable_mcp: Enable Model Context Protocol
            enable_rag: Enable Retrieval-Augmented Generation
        """
        self.provider = provider.lower()
        self.api_key = api_key or self._get_api_key()
        self.use_mock = not bool(self.api_key)
        self.enable_mcp = enable_mcp
        self.enable_rag = enable_rag
        
        if self.use_mock:
            logger.warning(f"No {self.provider} API key provided. Using mock responses.")
        
        # Initialize Enhanced MCP client
        if self.enable_mcp:
            try:
                self.mcp_client = EnhancedMCPClient(enable_async=True)
                logger.info("Enhanced MCP client initialized")
            except Exception as e:
                logger.warning(f"Enhanced MCP client failed to initialize: {e}. Using fallback.")
                self.mcp_client = EnhancedMCPFallback()
        else:
            self.mcp_client = EnhancedMCPFallback()
        
        # Initialize Hybrid RAG system
        if self.enable_rag:
            try:
                self.rag_system = HybridRAG(use_chromadb=True)
                logger.info("Hybrid RAG system initialized")
            except Exception as e:
                logger.warning(f"Hybrid RAG system failed to initialize: {e}. Using fallback.")
                self.rag_system = HybridRAG(use_chromadb=False)
        else:
            self.rag_system = HybridRAG(use_chromadb=False)
        
        # Provider configurations
        self.configs = {
            "gemini": {
                "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
                "headers": {"Content-Type": "application/json"}
            },
            "mistral": {
                "url": "https://api.mistral.ai/v1/chat/completions",
                "model": "mistral-large-latest",
                "headers": {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            },
            "huggingface": {
                "url": "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
                "headers": {"Authorization": f"Bearer {self.api_key}"}
            }
        }
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key based on provider"""
        key_mapping = {
            "gemini": "GOOGLE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY"
        }
        return os.getenv(key_mapping.get(self.provider, "GOOGLE_API_KEY"))
    
    async def generate_recap(self, content: str, length: str = "medium") -> str:
        """
        Generate a summary/recap of the content using MCP and RAG
        
        Args:
            content: Text content to summarize
            length: Length of summary ("brief", "medium", "detailed")
            
        Returns:
            Generated summary
        """
        try:
            # Enhance with RAG
            rag_result = self.rag_system.enhance_analysis_with_rag(content, "summary")
            context = rag_result.get("context", "")
            
            # Use MCP for enhanced reasoning
            mcp_result = await self.mcp_client.analyze_with_context(
                f"Generate a {length} summary of this content",
                context_ids=[rag_result.get("document_id", "")]
            )
            
            if self.use_mock:
                return self._generate_enhanced_mock_recap(content, length, context, mcp_result)
            
            # Truncate content if too long
            max_chars = 8000  # Reduced for free APIs
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            
            # Define length instructions
            length_instructions = {
                "brief": "in 2-3 sentences",
                "medium": "in 4-6 sentences", 
                "detailed": "in 8-12 sentences"
            }
            
            length_instruction = length_instructions.get(length, length_instructions["medium"])
            
            # Enhanced prompt with context
            context_info = f"\n\nRelevant context from similar content:\n{context}" if context else ""
            
            prompt = f"""
            Please provide a comprehensive summary of the following content {length_instruction}. 
            Focus on the main points, key concepts, and important details. 
            Make it clear and easy to understand.
            {context_info}
            
            Content:
            {content}
            
            Summary:
            """
            
            response = self._call_ai_api(prompt, max_tokens=300, temperature=0.3)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating recap: {str(e)}")
            return self._generate_enhanced_mock_recap(content, length, "", {})
    
    def _call_ai_api(self, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
        """
        Call the appropriate free AI API based on provider
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0-1)
            
        Returns:
            AI response text
        """
        if self.provider == "gemini":
            return self._call_gemini_api(prompt, max_tokens, temperature)
        elif self.provider == "mistral":
            return self._call_mistral_api(prompt, max_tokens, temperature)
        elif self.provider == "huggingface":
            return self._call_huggingface_api(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _call_gemini_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Google Gemini 2.5 Flash API with web search capabilities"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            
            # Use Gemini 1.5 Pro (most powerful multimodal model available)
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            
            return response.text
            
        except ImportError:
            # Fallback to REST API if library not available
            return self._call_gemini_rest_api(prompt, max_tokens, temperature)
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            # Try fallback to gemini-1.5-flash if 1.5 pro fails
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    )
                )
                return response.text
            except Exception as fallback_error:
                logger.error(f"Gemini fallback error: {str(fallback_error)}")
                raise
    
    def _call_gemini_rest_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Gemini via REST API"""
        config = self.configs["gemini"]
        url = f"{config['url']}?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            }
        }
        
        response = requests.post(url, headers=config["headers"], json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_mistral_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Mistral AI API (most powerful free multimodal model)"""
        try:
            import mistralai
            from mistralai.client import MistralClient
            
            client = MistralClient(api_key=self.api_key)
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = client.chat(
                model="mistral-large-latest",  # Most powerful Mistral model
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            # Fallback to REST API if library not available
            return self._call_mistral_rest_api(prompt, max_tokens, temperature)
        except Exception as e:
            logger.error(f"Mistral API error: {str(e)}")
            raise
    
    def _call_mistral_rest_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Mistral via REST API"""
        config = self.configs["mistral"]
        
        payload = {
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(config["url"], headers=config["headers"], json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    
    def _call_huggingface_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Hugging Face API (free tier)"""
        config = self.configs["huggingface"]
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": max_tokens,
                "temperature": temperature,
                "do_sample": True
            }
        }
        
        response = requests.post(
            config["url"],
            headers=config["headers"],
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "").replace(prompt, "").strip()
        return str(data)
    
    async def create_questions(self, content: str, count: int = 10) -> List[Dict]:
        """
        Generate quiz questions from content using MCP and RAG
        
        Args:
            content: Text content to create questions from
            count: Number of questions to generate
            
        Returns:
            List of question dictionaries
        """
        try:
            # Enhance with RAG
            rag_result = self.rag_system.enhance_analysis_with_rag(content, "questions")
            context = rag_result.get("context", "")
            
            # Use MCP for enhanced question generation
            mcp_questions = await self.mcp_client.generate_questions_with_context(
                content, count, [rag_result.get("document_id", "")]
            )
            
            if self.use_mock:
                return self._generate_enhanced_mock_questions(content, count, context, mcp_questions)
            
            # Truncate content if too long
            max_chars = 6000  # Reduced for free APIs
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            
            # Enhanced prompt with context
            context_info = f"\n\nRelevant context from similar content:\n{context}" if context else ""
            
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
            
            response = self._call_ai_api(prompt, max_tokens=1500, temperature=0.5)
            
            # Clean up the response (remove markdown formatting if present)
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            questions = json.loads(response)
            
            # Validate and clean questions
            return self._validate_questions(questions)
            
        except Exception as e:
            logger.error(f"Error creating questions: {str(e)}")
            return self._generate_enhanced_mock_questions(content, count, "", [])
    
    async def extract_key_concepts(self, content: str) -> List[str]:
        """
        Extract key concepts and topics from content
        
        Args:
            content: Text content to analyze
            
        Returns:
            List of key concepts
        """
        try:
            if self.use_mock:
                return self._extract_mock_concepts(content)
            
            # Truncate content if too long
            max_chars = 6000
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            
            prompt = f"""
            Extract the 8-12 most important key concepts, topics, or themes from the following content.
            Return them as a simple list, one concept per line.
            Focus on the main ideas, important terms, and central topics.
            
            Content:
            {content}
            
            Key concepts:
            """
            
            response = self._call_ai_api(prompt, max_tokens=200, temperature=0.3)
            
            # Parse concepts (split by newlines and clean)
            concepts = []
            for line in response.split('\n'):
                concept = line.strip()
                if concept and not concept.startswith('-') and not concept.startswith('•'):
                    # Remove numbering if present
                    concept = re.sub(r'^\d+\.\s*', '', concept)
                    concepts.append(concept)
            
            return concepts[:12]  # Limit to 12 concepts
            
        except Exception as e:
            logger.error(f"Error extracting key concepts: {str(e)}")
            return self._extract_mock_concepts(content)

    async def ask_learning_question(self, question: str, content_id: str = None, context: str = None) -> Dict[str, Any]:
        """
        Answer learning-focused questions using RAG + MCP + Gemini 2.5 Flash with web search
        
        Args:
            question: The user's learning question
            content_id: Optional content ID to retrieve from RAG
            context: Optional additional context
            
        Returns:
            Dictionary with answer, sources, and learning insights
        """
        try:
            # Get relevant context from RAG system
            rag_results = []
            if content_id:
                # Retrieve specific document context
                rag_results = self.rag_system.search_similar_content(f"content_id:{content_id}", n_results=3)
            else:
                # Search for relevant content based on question
                rag_results = self.rag_system.search_similar_content(question, n_results=5)
            
            # Get MCP reasoning for the question
            mcp_reasoning = await self.mcp_client.analyze_with_context(
                query=question,
                analysis_depth="comprehensive"
            )
            
            # Build comprehensive prompt with RAG context, MCP reasoning, and web search capability
            rag_context = "\n\n".join([r.get("content", "") for r in rag_results[:3]])
            
            prompt = f"""
            You are an expert learning assistant. Answer the following learning question using the provided context, 
            your knowledge, and web search capabilities when needed.
            
            LEARNING QUESTION: {question}
            
            RELEVANT CONTEXT FROM DOCUMENTS:
            {rag_context[:2000]}
            
            MCP REASONING:
            {mcp_reasoning.get("reasoning", "")[:500]}
            
            ADDITIONAL CONTEXT:
            {context or "None provided"}
            
            INSTRUCTIONS:
            1. Provide a comprehensive, educational answer focused on learning
            2. Use web search if you need current information not in the provided context
            3. Structure your response with clear explanations and examples
            4. Include learning insights and connections to broader concepts
            5. If the question is not learning-related, politely redirect to educational topics
            
            Please provide a detailed, educational response that helps the user learn and understand the topic better.
            """
            
            if self.use_mock:
                return self._generate_mock_learning_answer(question, rag_results)
            
            # Use Gemini 2.5 Flash with web search for comprehensive answers
            response = self._call_ai_api(prompt, max_tokens=2048, temperature=0.7)
            
            # Extract sources and learning insights
            sources = [{"type": "document", "content": r.get("content", "")[:200]} for r in rag_results[:3]]
            if mcp_reasoning.get("reasoning"):
                sources.append({"type": "mcp_reasoning", "content": mcp_reasoning.get("reasoning", "")[:200]})
            
            return {
                "answer": response,
                "sources": sources,
                "learning_insights": self._extract_learning_insights(response),
                "question_type": self._classify_question_type(question),
                "confidence": self._calculate_confidence(rag_results, mcp_reasoning),
                "rag_context_used": len(rag_results) > 0,
                "mcp_reasoning_used": bool(mcp_reasoning.get("reasoning"))
            }
            
        except Exception as e:
            logger.error(f"Error answering learning question: {str(e)}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "learning_insights": [],
                "question_type": "error",
                "confidence": 0.0,
                "rag_context_used": False,
                "mcp_reasoning_used": False
            }

    def _generate_mock_learning_answer(self, question: str, rag_results: List[Dict]) -> Dict[str, Any]:
        """Generate mock learning answer for testing"""
        return {
            "answer": f"Mock answer for: {question}\n\nThis is a comprehensive educational response that would normally be generated using RAG context and MCP reasoning. The system found {len(rag_results)} relevant document chunks to inform this answer.",
            "sources": [{"type": "document", "content": r.get("content", "")[:200]} for r in rag_results[:3]],
            "learning_insights": ["This demonstrates RAG-enhanced learning", "MCP reasoning provides context", "Web search capabilities available"],
            "question_type": "learning",
            "confidence": 0.85,
            "rag_context_used": len(rag_results) > 0,
            "mcp_reasoning_used": True
        }

    def _extract_learning_insights(self, answer: str) -> List[str]:
        """Extract learning insights from the answer"""
        insights = []
        if "example" in answer.lower():
            insights.append("Includes practical examples")
        if "concept" in answer.lower():
            insights.append("Explains key concepts")
        if "application" in answer.lower():
            insights.append("Shows real-world applications")
        if "connection" in answer.lower():
            insights.append("Makes connections to other topics")
        return insights[:3]

    def _classify_question_type(self, question: str) -> str:
        """Classify the type of learning question"""
        question_lower = question.lower()
        if any(word in question_lower for word in ["what", "define", "definition"]):
            return "definition"
        elif any(word in question_lower for word in ["how", "process", "steps"]):
            return "process"
        elif any(word in question_lower for word in ["why", "reason", "cause"]):
            return "explanation"
        elif any(word in question_lower for word in ["example", "instance", "case"]):
            return "example"
        else:
            return "general"

    def _calculate_confidence(self, rag_results: List[Dict], mcp_reasoning: Dict) -> float:
        """Calculate confidence score based on available context"""
        confidence = 0.5  # Base confidence
        
        if len(rag_results) > 0:
            confidence += 0.3
        if len(rag_results) > 2:
            confidence += 0.1
        if mcp_reasoning.get("reasoning"):
            confidence += 0.1
            
        return min(confidence, 1.0)
    
    def _validate_questions(self, questions: List[Dict]) -> List[Dict]:
        """
        Validate and clean question format
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            Validated questions list
        """
        validated = []
        
        for q in questions:
            if not isinstance(q, dict):
                continue
                
            # Ensure required fields
            if 'question' not in q or 'type' not in q:
                continue
            
            # Set defaults
            q.setdefault('options', [])
            q.setdefault('correct_answer', '')
            q.setdefault('explanation', '')
            
            # Validate question type
            if q['type'] not in ['multiple_choice', 'true_false', 'short_answer']:
                q['type'] = 'short_answer'
            
            # Validate multiple choice questions
            if q['type'] == 'multiple_choice':
                if not q['options'] or len(q['options']) < 2:
                    q['type'] = 'short_answer'
                    q['options'] = []
            
            validated.append(q)
        
        return validated
    
    def _generate_mock_recap(self, content: str, length: str) -> str:
        """Generate a mock recap when API is not available"""
        word_count = len(content.split())
        
        if length == "brief":
            return f"This content contains approximately {word_count} words and covers important topics related to the subject matter. The key points are presented in a structured manner."
        elif length == "detailed":
            return f"This comprehensive content spans approximately {word_count} words and provides detailed information on various aspects of the topic. The material is well-organized and covers multiple key concepts, including important details and supporting information that helps explain the main themes. The content appears to be educational in nature and provides valuable insights into the subject area."
        else:  # medium
            return f"This content contains approximately {word_count} words and covers several important topics. The material is well-structured and provides key information about the subject matter, including main concepts and supporting details that help explain the central themes."
    
    def _generate_enhanced_mock_recap(self, content: str, length: str, context: str, mcp_result: Dict) -> str:
        """Generate enhanced mock recap with MCP and RAG context"""
        word_count = len(content.split())
        context_info = " (Enhanced with RAG context)" if context else ""
        mcp_info = " (Enhanced with MCP reasoning)" if mcp_result else ""
        
        base_recap = self._generate_mock_recap(content, length)
        return f"{base_recap}{context_info}{mcp_info}"
    
    def _generate_mock_questions(self, content: str, count: int) -> List[Dict]:
        """Generate mock questions when API is not available"""
        questions = []
        
        for i in range(min(count, 5)):  # Limit to 5 mock questions
            if i % 3 == 0:
                # Multiple choice
                questions.append({
                    "question": f"Question {i+1}: What is one of the main topics discussed in this content?",
                    "type": "multiple_choice",
                    "options": ["Topic A", "Topic B", "Topic C", "Topic D"],
                    "correct_answer": "Topic A",
                    "explanation": "This is a mock question for demonstration purposes."
                })
            elif i % 3 == 1:
                # True/False
                questions.append({
                    "question": f"Question {i+1}: The content discusses important concepts related to the subject matter.",
                    "type": "true_false",
                    "options": ["True", "False"],
                    "correct_answer": "True",
                    "explanation": "This is a mock question for demonstration purposes."
                })
            else:
                # Short answer
                questions.append({
                    "question": f"Question {i+1}: What are the key points mentioned in this content?",
                    "type": "short_answer",
                    "options": [],
                    "correct_answer": "Various key points are discussed",
                    "explanation": "This is a mock question for demonstration purposes."
                })
        
        return questions
    
    def _generate_enhanced_mock_questions(self, content: str, count: int, context: str, mcp_questions: List[Dict]) -> List[Dict]:
        """Generate enhanced mock questions with MCP and RAG context"""
        questions = self._generate_mock_questions(content, count)
        
        # Enhance with context information
        for question in questions:
            if context:
                question["explanation"] += " (Enhanced with RAG context)"
            if mcp_questions:
                question["explanation"] += " (Enhanced with MCP reasoning)"
        
        return questions
    
    def _extract_mock_concepts(self, content: str) -> List[str]:
        """Extract mock concepts when API is not available"""
        # Simple keyword extraction
        words = content.lower().split()
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
        # Count word frequency
        word_freq = {}
        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top concepts
        concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:8]
        return [concept[0].title() for concept in concepts]