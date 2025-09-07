"""
Enhanced Quiz Generator
Creates dynamic, engaging quizzes with expandable questions
"""

import logging
import random
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedQuizGenerator:
    """Advanced quiz generator with multiple question types and expandable content"""
    
    def __init__(self):
        self.question_types = [
            "multiple_choice",
            "true_false", 
            "fill_in_blank",
            "short_answer",
            "matching",
            "essay"
        ]
        self.difficulty_levels = ["easy", "medium", "hard", "expert"]
        
    def generate_quiz(self, content: str, content_type: str = "text", 
                     min_questions: int = 10, max_questions: int = 20,
                     difficulty: str = "medium") -> Dict[str, Any]:
        """Generate a comprehensive quiz from content"""
        try:
            # Analyze content for quiz generation
            content_analysis = self._analyze_content_for_quiz(content, content_type)
            
            # Generate questions based on content
            questions = self._generate_questions(content, content_analysis, min_questions, difficulty)
            
            # Ensure we have at least min_questions
            while len(questions) < min_questions:
                additional_questions = self._generate_additional_questions(content, content_analysis, difficulty)
                questions.extend(additional_questions[:min_questions - len(questions)])
            
            # Limit to max_questions
            questions = questions[:max_questions]
            
            # Create quiz metadata
            quiz_metadata = self._create_quiz_metadata(content_analysis, len(questions), difficulty)
            
            return {
                "success": True,
                "quiz": {
                    "metadata": quiz_metadata,
                    "questions": questions,
                    "total_questions": len(questions),
                    "estimated_time": self._estimate_quiz_time(questions),
                    "difficulty_distribution": self._get_difficulty_distribution(questions)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _analyze_content_for_quiz(self, content: str, content_type: str) -> Dict[str, Any]:
        """Analyze content to determine quiz generation strategy"""
        try:
            # Basic content analysis
            words = content.split()
            sentences = content.split('.')
            
            # Extract key concepts
            key_concepts = self._extract_key_concepts(content)
            
            # Determine content complexity
            complexity = self._assess_content_complexity(content)
            
            # Identify question-worthy topics
            topics = self._identify_quiz_topics(content, key_concepts)
            
            return {
                "word_count": len(words),
                "sentence_count": len(sentences),
                "key_concepts": key_concepts,
                "complexity": complexity,
                "topics": topics,
                "content_type": content_type,
                "has_numbers": any(char.isdigit() for char in content),
                "has_dates": self._extract_dates(content),
                "has_definitions": self._find_definitions(content)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            return {"error": str(e)}
    
    def _generate_questions(self, content: str, analysis: Dict[str, Any], 
                          num_questions: int, difficulty: str) -> List[Dict[str, Any]]:
        """Generate questions based on content analysis"""
        questions = []
        
        try:
            # Generate different types of questions
            question_weights = self._get_question_type_weights(analysis, difficulty)
            
            for i in range(num_questions):
                question_type = self._select_question_type(question_weights)
                
                if question_type == "multiple_choice":
                    question = self._generate_multiple_choice(content, analysis, difficulty)
                elif question_type == "true_false":
                    question = self._generate_true_false(content, analysis, difficulty)
                elif question_type == "fill_in_blank":
                    question = self._generate_fill_in_blank(content, analysis, difficulty)
                elif question_type == "short_answer":
                    question = self._generate_short_answer(content, analysis, difficulty)
                elif question_type == "matching":
                    question = self._generate_matching(content, analysis, difficulty)
                else:  # essay
                    question = self._generate_essay(content, analysis, difficulty)
                
                if question:
                    question["id"] = i + 1
                    question["type"] = question_type
                    question["difficulty"] = difficulty
                    questions.append(question)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []
    
    def _generate_multiple_choice(self, content: str, analysis: Dict[str, Any], 
                                difficulty: str) -> Dict[str, Any]:
        """Generate multiple choice question"""
        try:
            # Extract a key concept or fact
            concepts = analysis.get("key_concepts", [])
            if not concepts:
                return None
            
            concept = random.choice(concepts)
            
            # Create question based on concept
            question_text = f"What is the main purpose of {concept}?"
            
            # Generate options
            correct_answer = f"The main purpose of {concept} is to provide essential functionality."
            wrong_answers = [
                f"{concept} is primarily used for decorative purposes.",
                f"The main function of {concept} is to slow down processes.",
                f"{concept} is only relevant in specific contexts."
            ]
            
            options = [correct_answer] + wrong_answers
            random.shuffle(options)
            
            return {
                "question": question_text,
                "options": options,
                "correct_answer": correct_answer,
                "explanation": f"This question tests understanding of {concept} and its primary function.",
                "points": self._get_points_for_difficulty(difficulty)
            }
            
        except Exception as e:
            logger.error(f"Error generating multiple choice: {str(e)}")
            return None
    
    def _generate_true_false(self, content: str, analysis: Dict[str, Any], 
                           difficulty: str) -> Dict[str, Any]:
        """Generate true/false question"""
        try:
            concepts = analysis.get("key_concepts", [])
            if not concepts:
                return None
            
            concept = random.choice(concepts)
            is_true = random.choice([True, False])
            
            if is_true:
                statement = f"{concept} is an important concept in this context."
                explanation = f"Correct! {concept} is indeed important as mentioned in the content."
            else:
                statement = f"{concept} is not relevant to this topic."
                explanation = f"Incorrect! {concept} is actually very relevant to this topic."
            
            return {
                "question": statement,
                "correct_answer": is_true,
                "explanation": explanation,
                "points": self._get_points_for_difficulty(difficulty)
            }
            
        except Exception as e:
            logger.error(f"Error generating true/false: {str(e)}")
            return None
    
    def _generate_fill_in_blank(self, content: str, analysis: Dict[str, Any], 
                              difficulty: str) -> Dict[str, Any]:
        """Generate fill-in-the-blank question"""
        try:
            concepts = analysis.get("key_concepts", [])
            if not concepts:
                return None
            
            concept = random.choice(concepts)
            
            # Create a sentence with a blank
            question_text = f"The concept of _____ is crucial for understanding this topic."
            correct_answer = concept
            
            return {
                "question": question_text,
                "correct_answer": correct_answer,
                "explanation": f"The blank should be filled with '{concept}' as it's a key concept in the content.",
                "points": self._get_points_for_difficulty(difficulty)
            }
            
        except Exception as e:
            logger.error(f"Error generating fill-in-blank: {str(e)}")
            return None
    
    def _generate_short_answer(self, content: str, analysis: Dict[str, Any], 
                             difficulty: str) -> Dict[str, Any]:
        """Generate short answer question"""
        try:
            concepts = analysis.get("key_concepts", [])
            if not concepts:
                return None
            
            concept = random.choice(concepts)
            
            question_text = f"Explain the significance of {concept} in this context."
            
            return {
                "question": question_text,
                "correct_answer": f"{concept} is significant because it plays a crucial role in the overall understanding of the topic.",
                "explanation": f"This question requires you to explain why {concept} is important in the given context.",
                "points": self._get_points_for_difficulty(difficulty) * 2  # Short answers worth more points
            }
            
        except Exception as e:
            logger.error(f"Error generating short answer: {str(e)}")
            return None
    
    def _generate_matching(self, content: str, analysis: Dict[str, Any], 
                         difficulty: str) -> Dict[str, Any]:
        """Generate matching question"""
        try:
            concepts = analysis.get("key_concepts", [])
            if len(concepts) < 4:
                return None
            
            # Select 4 concepts for matching
            selected_concepts = random.sample(concepts, 4)
            
            # Create definitions
            definitions = [f"Definition for {concept}" for concept in selected_concepts]
            random.shuffle(definitions)
            
            return {
                "question": "Match each concept with its definition:",
                "left_items": selected_concepts,
                "right_items": definitions,
                "correct_matches": {concept: f"Definition for {concept}" for concept in selected_concepts},
                "explanation": "This matching exercise tests your understanding of key concepts and their definitions.",
                "points": self._get_points_for_difficulty(difficulty) * 1.5
            }
            
        except Exception as e:
            logger.error(f"Error generating matching: {str(e)}")
            return None
    
    def _generate_essay(self, content: str, analysis: Dict[str, Any], 
                       difficulty: str) -> Dict[str, Any]:
        """Generate essay question"""
        try:
            concepts = analysis.get("key_concepts", [])
            if not concepts:
                return None
            
            main_concept = random.choice(concepts)
            
            question_text = f"Write a comprehensive essay discussing the role and importance of {main_concept}. Include examples and explain how it relates to other concepts in the content."
            
            return {
                "question": question_text,
                "correct_answer": f"Your essay should discuss {main_concept} in detail, providing examples and connections to other concepts.",
                "explanation": f"This essay question requires deep understanding of {main_concept} and its relationships with other concepts.",
                "points": self._get_points_for_difficulty(difficulty) * 3,  # Essays worth most points
                "word_limit": 300 if difficulty == "easy" else 500 if difficulty == "medium" else 750
            }
            
        except Exception as e:
            logger.error(f"Error generating essay: {str(e)}")
            return None
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""
        try:
            # Simple key concept extraction
            words = content.lower().split()
            word_freq = {}
            
            for word in words:
                if len(word) > 4:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top concepts
            concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            return [concept[0] for concept in concepts]
            
        except Exception as e:
            logger.error(f"Error extracting key concepts: {str(e)}")
            return []
    
    def _assess_content_complexity(self, content: str) -> str:
        """Assess content complexity"""
        try:
            words = content.split()
            sentences = content.split('.')
            
            avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
            
            if avg_words_per_sentence < 10:
                return "easy"
            elif avg_words_per_sentence < 20:
                return "medium"
            else:
                return "hard"
                
        except Exception as e:
            logger.error(f"Error assessing complexity: {str(e)}")
            return "medium"
    
    def _identify_quiz_topics(self, content: str, concepts: List[str]) -> List[str]:
        """Identify topics suitable for quiz questions"""
        return concepts[:5]  # Use top 5 concepts
    
    def _get_question_type_weights(self, analysis: Dict[str, Any], difficulty: str) -> Dict[str, float]:
        """Get weights for different question types based on content and difficulty"""
        base_weights = {
            "multiple_choice": 0.3,
            "true_false": 0.2,
            "fill_in_blank": 0.15,
            "short_answer": 0.2,
            "matching": 0.1,
            "essay": 0.05
        }
        
        # Adjust weights based on difficulty
        if difficulty == "easy":
            base_weights["multiple_choice"] += 0.1
            base_weights["true_false"] += 0.1
            base_weights["essay"] -= 0.05
        elif difficulty == "hard":
            base_weights["short_answer"] += 0.1
            base_weights["essay"] += 0.1
            base_weights["multiple_choice"] -= 0.05
        
        return base_weights
    
    def _select_question_type(self, weights: Dict[str, float]) -> str:
        """Select question type based on weights"""
        types = list(weights.keys())
        probabilities = list(weights.values())
        
        return random.choices(types, weights=probabilities)[0]
    
    def _get_points_for_difficulty(self, difficulty: str) -> int:
        """Get points for question based on difficulty"""
        points_map = {
            "easy": 1,
            "medium": 2,
            "hard": 3,
            "expert": 5
        }
        return points_map.get(difficulty, 2)
    
    def _create_quiz_metadata(self, analysis: Dict[str, Any], num_questions: int, difficulty: str) -> Dict[str, Any]:
        """Create quiz metadata"""
        return {
            "title": f"Quiz on {analysis.get('content_type', 'Content')}",
            "description": f"Comprehensive quiz with {num_questions} questions",
            "difficulty": difficulty,
            "created_at": datetime.now().isoformat(),
            "estimated_time": num_questions * 2,  # 2 minutes per question
            "total_points": num_questions * self._get_points_for_difficulty(difficulty),
            "topics_covered": analysis.get("topics", [])[:5]
        }
    
    def _estimate_quiz_time(self, questions: List[Dict[str, Any]]) -> int:
        """Estimate total quiz time in minutes"""
        time_map = {
            "multiple_choice": 1,
            "true_false": 0.5,
            "fill_in_blank": 1,
            "short_answer": 3,
            "matching": 2,
            "essay": 10
        }
        
        total_time = sum(time_map.get(q.get("type", "multiple_choice"), 1) for q in questions)
        return int(total_time)
    
    def _get_difficulty_distribution(self, questions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of question difficulties"""
        distribution = {}
        for question in questions:
            diff = question.get("difficulty", "medium")
            distribution[diff] = distribution.get(diff, 0) + 1
        return distribution
    
    def _extract_dates(self, content: str) -> List[str]:
        """Extract dates from content"""
        import re
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b'
        return re.findall(date_pattern, content)
    
    def _find_definitions(self, content: str) -> List[str]:
        """Find definition patterns in content"""
        import re
        definition_pattern = r'(\w+)\s+is\s+(?:a|an|the)?\s+([^.]*)'
        return re.findall(definition_pattern, content, re.IGNORECASE)
    
    def _generate_additional_questions(self, content: str, analysis: Dict[str, Any], 
                                     difficulty: str) -> List[Dict[str, Any]]:
        """Generate additional questions when needed"""
        return self._generate_questions(content, analysis, 5, difficulty)

