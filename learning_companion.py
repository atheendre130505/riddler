"""
Learning Companion - Conversational AI for Educational Discussions
Focuses on discussion, exploration, and making learning fun
"""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningCompanion:
    """AI Learning Companion that treats students as equals and focuses on discussion"""
    
    def __init__(self):
        self.conversation_history = []
        self.student_profile = {}
        self.learning_context = {}
        self.personality_traits = {
            "encouraging": True,
            "curious": True,
            "patient": True,
            "enthusiastic": True,
            "supportive": True
        }
        
        # Conversation starters and responses
        self.conversation_starters = [
            "That's a great question! Let me think about this with you...",
            "I love your curiosity! This is exactly the kind of thinking that leads to breakthroughs.",
            "You know what? That's a really interesting perspective. Let's explore it together.",
            "I'm excited to dive into this with you! What made you think about this?",
            "That's such a thoughtful question! I can tell you're really engaging with the material."
        ]
        
        self.encouraging_responses = [
            "You're absolutely on the right track!",
            "That's brilliant thinking!",
            "I can see you're really getting it!",
            "You're asking exactly the right questions!",
            "This is the kind of deep thinking that makes learning exciting!",
            "You're making connections that many people miss!",
            "I love how you're thinking about this!",
            "You're developing such a strong understanding!"
        ]
        
        self.curiosity_prompts = [
            "What do you think would happen if...?",
            "I'm curious about your thoughts on...",
            "What's your take on...?",
            "How do you see this connecting to...?",
            "What questions does this raise for you?",
            "I wonder what you think about...",
            "What's your perspective on...?",
            "How would you explain this to someone else?"
        ]
    
    def start_conversation(self, topic: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start a new learning conversation"""
        try:
            self.learning_context = context or {}
            self.conversation_history = []
            
            # Create welcoming response
            welcome_message = self._create_welcome_message(topic)
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "message": welcome_message,
                "timestamp": datetime.now().isoformat(),
                "type": "welcome"
            })
            
            return {
                "success": True,
                "message": welcome_message,
                "conversation_id": f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "context": self.learning_context
            }
            
        except Exception as e:
            logger.error(f"Error starting conversation: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def respond_to_question(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Respond to student's question with engaging, educational discussion"""
        try:
            # Update context
            if context:
                self.learning_context.update(context)
            
            # Analyze the question
            question_analysis = self._analyze_question(question)
            
            # Generate response based on question type and context
            response = self._generate_engaging_response(question, question_analysis)
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "message": question,
                "timestamp": datetime.now().isoformat(),
                "type": "question",
                "analysis": question_analysis
            })
            
            self.conversation_history.append({
                "role": "assistant",
                "message": response["message"],
                "timestamp": datetime.now().isoformat(),
                "type": "response",
                "suggestions": response.get("suggestions", []),
                "follow_up": response.get("follow_up", None)
            })
            
            return {
                "success": True,
                "message": response["message"],
                "suggestions": response.get("suggestions", []),
                "follow_up": response.get("follow_up", None),
                "conversation_context": self._get_conversation_summary()
            }
            
        except Exception as e:
            logger.error(f"Error responding to question: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_welcome_message(self, topic: str) -> str:
        """Create a welcoming message for the conversation"""
        welcome_templates = [
            f"Hey there! I'm so excited to explore {topic} with you! I love how you're diving into this topic. What's got you most curious about it?",
            f"Welcome! I'm thrilled to be your learning companion for {topic}. I can already tell you're going to ask some amazing questions. What's on your mind?",
            f"Hi! I'm here to learn alongside you about {topic}. I think the best learning happens when we discuss and explore together. What would you like to dive into first?",
            f"Hello! I'm really looking forward to our discussion about {topic}. I love how you're approaching this with such curiosity. What's your first question?",
            f"Hey! I'm excited to be your learning buddy for {topic}. I believe the best learning happens through conversation and exploration. What's got you thinking?"
        ]
        
        return random.choice(welcome_templates)
    
    def _analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze the student's question to understand intent and complexity"""
        try:
            question_lower = question.lower()
            
            # Determine question type
            question_type = "general"
            if any(word in question_lower for word in ["what", "why", "how", "when", "where", "who"]):
                question_type = "inquiry"
            elif any(word in question_lower for word in ["explain", "describe", "tell me about"]):
                question_type = "explanation"
            elif any(word in question_lower for word in ["difference", "compare", "contrast"]):
                question_type = "comparison"
            elif any(word in question_lower for word in ["example", "instance", "case"]):
                question_type = "example"
            elif any(word in question_lower for word in ["problem", "issue", "challenge"]):
                question_type = "problem_solving"
            
            # Determine complexity
            complexity = "medium"
            if len(question.split()) > 20:
                complexity = "high"
            elif len(question.split()) < 5:
                complexity = "low"
            
            # Extract key concepts
            key_concepts = self._extract_concepts_from_question(question)
            
            # Determine emotional tone
            tone = "neutral"
            if any(word in question_lower for word in ["confused", "don't understand", "stuck"]):
                tone = "confused"
            elif any(word in question_lower for word in ["excited", "love", "fascinating"]):
                tone = "excited"
            elif any(word in question_lower for word in ["frustrated", "difficult", "hard"]):
                tone = "frustrated"
            
            return {
                "type": question_type,
                "complexity": complexity,
                "key_concepts": key_concepts,
                "tone": tone,
                "word_count": len(question.split())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing question: {str(e)}")
            return {"type": "general", "complexity": "medium", "key_concepts": [], "tone": "neutral"}
    
    def _generate_engaging_response(self, question: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an engaging, educational response"""
        try:
            question_type = analysis.get("type", "general")
            complexity = analysis.get("complexity", "medium")
            tone = analysis.get("tone", "neutral")
            
            # Start with encouraging response
            response_parts = [random.choice(self.conversation_starters)]
            
            # Generate main response based on question type
            if question_type == "inquiry":
                main_response = self._generate_inquiry_response(question, analysis)
            elif question_type == "explanation":
                main_response = self._generate_explanation_response(question, analysis)
            elif question_type == "comparison":
                main_response = self._generate_comparison_response(question, analysis)
            elif question_type == "example":
                main_response = self._generate_example_response(question, analysis)
            elif question_type == "problem_solving":
                main_response = self._generate_problem_solving_response(question, analysis)
            else:
                main_response = self._generate_general_response(question, analysis)
            
            response_parts.append(main_response)
            
            # Add encouraging note
            if tone == "confused":
                response_parts.append("Don't worry, this is exactly how learning works! Every expert was once confused about these same concepts.")
            elif tone == "frustrated":
                response_parts.append("I can see you're working hard on this, and that's exactly what it takes to master new concepts!")
            else:
                response_parts.append(random.choice(self.encouraging_responses))
            
            # Generate follow-up questions
            follow_up = self._generate_follow_up_questions(question, analysis)
            
            # Generate suggestions
            suggestions = self._generate_learning_suggestions(question, analysis)
            
            return {
                "message": " ".join(response_parts),
                "follow_up": follow_up,
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "message": "That's a great question! I'm here to help you explore this topic together. What specific aspect would you like to dive into?",
                "follow_up": "What's your current understanding of this concept?",
                "suggestions": ["Let's break this down step by step", "What examples come to mind?"]
            }
    
    def _generate_inquiry_response(self, question: str, analysis: Dict[str, Any]) -> str:
        """Generate response for inquiry questions"""
        responses = [
            "That's such a thoughtful question! Let me share what I know and then I'd love to hear your thoughts on it.",
            "I love how you're thinking about this! Here's my perspective, and I'm curious about yours too.",
            "Great question! This is something that really gets me thinking. Let me share what I understand and then we can explore it together.",
            "You're asking exactly the right kind of question! This is how deep understanding develops. Let me share what I know..."
        ]
        return random.choice(responses)
    
    def _generate_explanation_response(self, question: str, analysis: Dict[str, Any]) -> str:
        """Generate response for explanation requests"""
        responses = [
            "I'd be happy to explain this! Let me break it down in a way that makes sense, and feel free to ask questions as we go.",
            "Absolutely! I love explaining concepts. Let me walk through this step by step, and you can tell me if anything needs clarification.",
            "Of course! This is such an important concept to understand. Let me explain it in a way that connects to what you already know.",
            "I'm excited to explain this! I'll start with the basics and build up, and you can jump in with questions anytime."
        ]
        return random.choice(responses)
    
    def _generate_comparison_response(self, question: str, analysis: Dict[str, Any]) -> str:
        """Generate response for comparison questions"""
        responses = [
            "That's a great way to think about it! Comparing concepts really helps deepen understanding. Let me walk through the similarities and differences with you.",
            "I love comparison questions! They really help us see the bigger picture. Let me break down how these concepts relate to each other.",
            "Excellent question! Comparing these concepts will really help clarify both of them. Let me share what I see as the key differences and similarities.",
            "This is such a smart approach! Let me help you see how these concepts connect and where they differ."
        ]
        return random.choice(responses)
    
    def _generate_example_response(self, question: str, analysis: Dict[str, Any]) -> str:
        """Generate response for example requests"""
        responses = [
            "I love examples! They make everything so much clearer. Let me give you a few examples and then I'd love to hear what examples you can think of.",
            "Great idea! Examples are the best way to understand concepts. Let me share some examples and then we can explore more together.",
            "Absolutely! Examples really bring concepts to life. Let me give you some concrete examples and then we can brainstorm more.",
            "I'm excited to share examples! They make everything click. Let me give you some examples and then you can share your own."
        ]
        return random.choice(responses)
    
    def _generate_problem_solving_response(self, question: str, analysis: Dict[str, Any]) -> str:
        """Generate response for problem-solving questions"""
        responses = [
            "I love problem-solving together! Let's work through this step by step. What's your first instinct about how to approach this?",
            "Great problem! Let's tackle this together. I'll guide you through my thinking process, and you can share yours too.",
            "This is exactly the kind of challenge that builds real understanding! Let me walk through my approach and then we can discuss yours.",
            "I'm excited to work through this with you! Let's break it down together and see what we discover."
        ]
        return random.choice(responses)
    
    def _generate_general_response(self, question: str, analysis: Dict[str, Any]) -> str:
        """Generate response for general questions"""
        responses = [
            "That's such an interesting question! Let me share what I think about this and then I'd love to hear your perspective.",
            "I love how you're thinking about this! Let me share my thoughts and then we can explore it together.",
            "Great question! This is something I find really fascinating. Let me share what I know and then we can discuss it further.",
            "You're asking exactly the kind of question that leads to deep understanding! Let me share my perspective..."
        ]
        return random.choice(responses)
    
    def _generate_follow_up_questions(self, question: str, analysis: Dict[str, Any]) -> str:
        """Generate follow-up questions to keep the conversation going"""
        concepts = analysis.get("key_concepts", [])
        
        if concepts:
            concept = random.choice(concepts)
            follow_ups = [
                f"What do you think about {concept}?",
                f"How does {concept} connect to what you already know?",
                f"What questions does {concept} raise for you?",
                f"How would you explain {concept} to someone else?"
            ]
        else:
            follow_ups = [
                "What's your current understanding of this?",
                "What made you think about this?",
                "How does this connect to what you already know?",
                "What would you like to explore further?"
            ]
        
        return random.choice(follow_ups)
    
    def _generate_learning_suggestions(self, question: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate learning suggestions based on the question"""
        suggestions = [
            "Let's break this down step by step",
            "What examples come to mind?",
            "How does this connect to what you already know?",
            "What would you like to explore further?",
            "Let's think about this from a different angle",
            "What questions does this raise for you?"
        ]
        
        # Add specific suggestions based on question type
        question_type = analysis.get("type", "general")
        if question_type == "comparison":
            suggestions.extend([
                "Let's create a comparison chart",
                "What similarities do you see?",
                "What differences stand out to you?"
            ])
        elif question_type == "problem_solving":
            suggestions.extend([
                "Let's work through this together",
                "What's your first step?",
                "What information do we need?"
            ])
        
        return random.sample(suggestions, min(3, len(suggestions)))
    
    def _extract_concepts_from_question(self, question: str) -> List[str]:
        """Extract key concepts from the question"""
        # Simple concept extraction
        words = question.lower().split()
        concepts = []
        
        # Look for capitalized words (potential proper nouns/concepts)
        for word in words:
            if word[0].isupper() and len(word) > 3:
                concepts.append(word)
        
        # Look for quoted terms
        import re
        quoted_terms = re.findall(r'"([^"]*)"', question)
        concepts.extend(quoted_terms)
        
        return concepts[:3]  # Return top 3 concepts
    
    def _get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation"""
        return {
            "message_count": len(self.conversation_history),
            "topics_discussed": list(set(
                concept for msg in self.conversation_history 
                if msg.get("analysis", {}).get("key_concepts")
                for concept in msg["analysis"]["key_concepts"]
            )),
            "conversation_duration": self._calculate_conversation_duration(),
            "learning_progress": self._assess_learning_progress()
        }
    
    def _calculate_conversation_duration(self) -> int:
        """Calculate conversation duration in minutes"""
        if len(self.conversation_history) < 2:
            return 0
        
        start_time = datetime.fromisoformat(self.conversation_history[0]["timestamp"])
        end_time = datetime.fromisoformat(self.conversation_history[-1]["timestamp"])
        return int((end_time - start_time).total_seconds() / 60)
    
    def _assess_learning_progress(self) -> str:
        """Assess learning progress based on conversation"""
        if len(self.conversation_history) < 4:
            return "just_started"
        elif len(self.conversation_history) < 8:
            return "exploring"
        elif len(self.conversation_history) < 12:
            return "developing"
        else:
            return "deep_diving"
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get full conversation history"""
        return self.conversation_history
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.learning_context = {}

