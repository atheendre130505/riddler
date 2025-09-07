"""
Enhanced Document Processor with Image Analysis
Handles text documents, images, and provides comprehensive analysis
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import base64
from PIL import Image
import io
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDocumentProcessor:
    """Enhanced processor for documents and images with comprehensive analysis"""
    
    def __init__(self):
        self.supported_text_formats = ['.pdf', '.docx', '.txt', '.md']
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        self.max_image_size = (1024, 1024)  # Resize large images
        
    def process_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process any document type and return comprehensive analysis"""
        try:
            file_ext = Path(filename).suffix.lower()
            
            if file_ext in self.supported_text_formats:
                return self._process_text_document(file_content, filename)
            elif file_ext in self.supported_image_formats:
                return self._process_image_document(file_content, filename)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file format: {file_ext}",
                    "supported_formats": self.supported_text_formats + self.supported_image_formats
                }
                
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "filename": filename
            }
    
    def _process_text_document(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Process text-based documents"""
        try:
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.pdf':
                text_content = self._extract_pdf_text(content)
            elif file_ext == '.docx':
                text_content = self._extract_docx_text(content)
            elif file_ext in ['.txt', '.md']:
                text_content = content.decode('utf-8')
            else:
                text_content = content.decode('utf-8', errors='ignore')
            
            # Analyze text content
            analysis = self._analyze_text_content(text_content, filename)
            
            return {
                "success": True,
                "type": "text",
                "filename": filename,
                "content": text_content,
                "analysis": analysis,
                "word_count": len(text_content.split()),
                "char_count": len(text_content)
            }
            
        except Exception as e:
            logger.error(f"Error processing text document {filename}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "filename": filename
            }
    
    def _process_image_document(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Process image documents"""
        try:
            # Load and process image
            image = Image.open(io.BytesIO(content))
            
            # Resize if too large
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            # Convert to base64 for analysis
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Analyze image content
            analysis = self._analyze_image_content(image, filename)
            
            return {
                "success": True,
                "type": "image",
                "filename": filename,
                "image_base64": img_base64,
                "image_size": image.size,
                "analysis": analysis,
                "format": image.format
            }
            
        except Exception as e:
            logger.error(f"Error processing image document {filename}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "filename": filename
            }
    
    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF"""
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return ""
    
    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX"""
        try:
            from docx import Document
            doc = Document(io.BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
            return ""
    
    def _analyze_text_content(self, text: str, filename: str) -> Dict[str, Any]:
        """Analyze text content for learning insights"""
        try:
            # Basic text analysis
            sentences = text.split('.')
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # Extract potential topics (simple keyword extraction)
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top topics
            top_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Estimate reading level (simple heuristic)
            avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
            reading_level = "Beginner" if avg_words_per_sentence < 10 else "Intermediate" if avg_words_per_sentence < 20 else "Advanced"
            
            return {
                "topics": [topic[0] for topic in top_topics],
                "topic_frequencies": dict(top_topics),
                "reading_level": reading_level,
                "sentence_count": len(sentences),
                "paragraph_count": len(paragraphs),
                "complexity_score": min(100, avg_words_per_sentence * 5),
                "key_phrases": self._extract_key_phrases(text),
                "estimated_reading_time": len(words) // 200  # words per minute
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text content: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_image_content(self, image: Image.Image, filename: str) -> Dict[str, Any]:
        """Analyze image content for learning insights"""
        try:
            # Basic image analysis
            width, height = image.size
            aspect_ratio = width / height if height > 0 else 1
            
            # Determine image type based on characteristics
            image_type = "diagram" if aspect_ratio > 1.5 else "chart" if aspect_ratio < 0.7 else "photo"
            
            # Color analysis
            colors = image.getcolors(maxcolors=256*256*256)
            if colors:
                dominant_color = max(colors, key=lambda x: x[0])[1]
                color_diversity = len(colors)
            else:
                dominant_color = (128, 128, 128)
                color_diversity = 0
            
            return {
                "image_type": image_type,
                "aspect_ratio": round(aspect_ratio, 2),
                "color_diversity": color_diversity,
                "dominant_color": dominant_color,
                "estimated_complexity": min(100, color_diversity // 100),
                "suggested_topics": self._suggest_image_topics(image_type, aspect_ratio)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image content: {str(e)}")
            return {"error": str(e)}
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        try:
            # Simple key phrase extraction
            sentences = text.split('.')
            phrases = []
            
            for sentence in sentences:
                words = sentence.strip().split()
                if len(words) >= 3:  # Only phrases with 3+ words
                    # Extract 2-3 word phrases
                    for i in range(len(words) - 1):
                        phrase = ' '.join(words[i:i+2])
                        if len(phrase) > 10:  # Filter short phrases
                            phrases.append(phrase)
            
            # Return most common phrases
            phrase_freq = {}
            for phrase in phrases:
                phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1
            
            return [phrase for phrase, freq in sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:5]]
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {str(e)}")
            return []
    
    def _suggest_image_topics(self, image_type: str, aspect_ratio: float) -> List[str]:
        """Suggest topics based on image characteristics"""
        if image_type == "diagram":
            return ["Process Flow", "System Architecture", "Workflow", "Technical Diagram"]
        elif image_type == "chart":
            return ["Data Visualization", "Statistics", "Graphs", "Charts"]
        else:
            return ["Visual Content", "Illustration", "Photograph", "Image Analysis"]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "supported_text_formats": self.supported_text_formats,
            "supported_image_formats": self.supported_image_formats,
            "max_image_size": self.max_image_size,
            "processor_version": "2.0.0"
        }

