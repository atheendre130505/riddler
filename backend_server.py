"""
Standalone Backend Server for Smart Content Agent
This can be deployed separately (Railway, Render, etc.) and called from Firebase Hosting
"""

import os
import json
import logging
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Content Agent API",
    description="AI-Powered Content Analysis Platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Firebase Hosting domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for lazy loading
_rag_system = None
_analyzer = None
_file_processor = None
_enhanced_processor = None
_quiz_generator = None
_learning_companion = None
_scraper = None

def get_rag_system():
    """Lazy load RAG system"""
    global _rag_system
    if _rag_system is None:
        try:
            from advanced_rag import AdvancedRAG
            _rag_system = AdvancedRAG()
            logger.info("RAG system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            _rag_system = None
    return _rag_system

def get_analyzer():
    """Lazy load analyzer"""
    global _analyzer
    if _analyzer is None:
        try:
            from enhanced_analyzer import EnhancedContentAnalyzer
            _analyzer = EnhancedContentAnalyzer(provider="gemini")
            logger.info("Content analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize analyzer: {e}")
            _analyzer = None
    return _analyzer

def get_file_processor():
    """Lazy load file processor"""
    global _file_processor
    if _file_processor is None:
        try:
            from processor import FileProcessor
            _file_processor = FileProcessor()
            logger.info("File processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize file processor: {e}")
            _file_processor = None
    return _file_processor

def get_enhanced_processor():
    """Lazy load enhanced processor"""
    global _enhanced_processor
    if _enhanced_processor is None:
        try:
            from enhanced_processor import EnhancedDocumentProcessor
            _enhanced_processor = EnhancedDocumentProcessor()
            logger.info("Enhanced processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced processor: {e}")
            _enhanced_processor = None
    return _enhanced_processor

def get_quiz_generator():
    """Lazy load quiz generator"""
    global _quiz_generator
    if _quiz_generator is None:
        try:
            from quiz_generator import EnhancedQuizGenerator
            _quiz_generator = EnhancedQuizGenerator()
            logger.info("Quiz generator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize quiz generator: {e}")
            _quiz_generator = None
    return _quiz_generator

def get_learning_companion():
    """Lazy load learning companion"""
    global _learning_companion
    if _learning_companion is None:
        try:
            from learning_companion import LearningCompanion
            _learning_companion = LearningCompanion()
            logger.info("Learning companion initialized")
        except Exception as e:
            logger.error(f"Failed to initialize learning companion: {e}")
            _learning_companion = None
    return _learning_companion

def get_scraper():
    """Lazy load scraper"""
    global _scraper
    if _scraper is None:
        try:
            from scraper import WebScraper
            _scraper = WebScraper()
            logger.info("Web scraper initialized")
        except Exception as e:
            logger.error(f"Failed to initialize scraper: {e}")
            _scraper = None
    return _scraper

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Content Agent API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "process_url": "/process-url",
            "ask_question": "/ask-question",
            "stats": "/stats",
            "enhanced_upload": "/enhanced-upload",
            "generate_quiz": "/generate-quiz",
            "start_conversation": "/start-conversation",
            "chat": "/chat"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Smart Content Agent API is running",
        "timestamp": str(asyncio.get_event_loop().time())
    }

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    provider: str = Form("gemini")
):
    """Upload and process a file"""
    try:
        logger.info(f"Processing file: {file.filename}")
        
        # Get file processor
        processor = get_file_processor()
        if not processor:
            raise HTTPException(status_code=500, detail="File processor not available")
        
        # Read file content
        content = await file.read()
        
        # Create temporary file for processing
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process file using FileProcessor
            result = processor.process_file(temp_file_path)
            processed_content = result["content"]
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
        
        # Get RAG system and analyzer
        rag = get_rag_system()
        analyzer = get_analyzer()
        
        if not rag or not analyzer:
            # Return basic processing result
            return {
                "status": "success",
                "filename": file.filename,
                "content_length": len(processed_content),
                "message": "File processed successfully (AI features not available)",
                "content_preview": processed_content[:500] + "..." if len(processed_content) > 500 else processed_content
            }
        
        # Store in RAG system
        try:
            rag.add_document(processed_content, {"filename": file.filename, "type": "upload"})
            logger.info(f"Document stored in RAG system: {file.filename}")
        except Exception as e:
            logger.error(f"Failed to store document in RAG: {e}")
        
        # Generate analysis using the actual content
        try:
            logger.info("Generating AI summary...")
            summary = await analyzer.enhanced_generate_recap(processed_content)
            logger.info("Summary generated successfully")
            
            logger.info("Extracting key concepts...")
            concepts = await analyzer.enhanced_extract_key_concepts(processed_content)
            logger.info("Key concepts extracted successfully")
            
            # Generate quiz using the actual content
            quiz_generator = get_quiz_generator()
            quiz_result = None
            if quiz_generator:
                try:
                    logger.info("Generating quiz from content...")
                    quiz_result = quiz_generator.generate_quiz(
                        processed_content, 
                        "text",
                        min_questions=10,
                        max_questions=20
                    )
                    logger.info("Quiz generated successfully")
                except Exception as e:
                    logger.warning(f"Quiz generation failed: {e}")
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            summary = f"AI analysis failed: {str(e)}"
            concepts = ["Content processed successfully"]
            quiz_result = None
        
        return {
            "status": "success",
            "filename": file.filename,
            "content_length": len(processed_content),
            "summary": summary,
            "key_concepts": concepts,
            "quiz": quiz_result.get("quiz") if quiz_result and quiz_result.get("success") else None,
            "message": "File processed and analyzed successfully with AI"
        }
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-url")
async def process_url(
    url: str = Form(...),
    provider: str = Form("gemini")
):
    """Process a URL and extract content"""
    try:
        logger.info(f"Processing URL: {url}")
        
        # Get scraper
        scraper = get_scraper()
        if not scraper:
            raise HTTPException(status_code=500, detail="Web scraper not available")
        
        # Scrape content
        content = await scraper.scrape_url(url)
        
        if not content:
            raise HTTPException(status_code=400, detail="Failed to extract content from URL")
        
        # Get RAG system and analyzer
        rag = get_rag_system()
        analyzer = get_analyzer()
        
        if not rag or not analyzer:
            # Return basic processing result
            return {
                "status": "success",
                "url": url,
                "content_length": len(content),
                "message": "URL processed successfully (AI features not available)",
                "content_preview": content[:500] + "..." if len(content) > 500 else content
            }
        
        # Store in RAG system
        rag.add_document(content, {"url": url, "type": "url"})
        
        # Generate analysis
        try:
            summary = await analyzer.enhanced_generate_recap(content)
            concepts = await analyzer.enhanced_extract_key_concepts(content)
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            summary = "AI analysis not available"
            concepts = ["Content processed successfully"]
        
        return {
            "status": "success",
            "url": url,
            "content_length": len(content),
            "summary": summary,
            "key_concepts": concepts,
            "message": "URL processed and analyzed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question")
async def ask_question(
    question: str = Form(...),
    session_id: str = Form("default")
):
    """Ask a question about the content"""
    try:
        logger.info(f"Processing question: {question}")
        
        # Get analyzer
        analyzer = get_analyzer()
        if not analyzer:
            raise HTTPException(status_code=500, detail="Content analyzer not available")
        
        # Get RAG system for context
        rag = get_rag_system()
        if not rag:
            raise HTTPException(status_code=500, detail="RAG system not available")
        
        # Search for relevant content using RAG
        try:
            logger.info(f"Searching for relevant content for question: {question[:50]}...")
            search_results = rag.hybrid_search(question, n_results=5)
            context = "\n".join([doc["content"] for doc in search_results])
            logger.info(f"Found {len(search_results)} relevant documents")
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            context = "No relevant content found"
        
        # Generate answer using the context from RAG
        try:
            logger.info("Generating AI response...")
            answer = await analyzer.enhanced_ask_learning_question(question, context, session_id)
            logger.info("AI response generated successfully")
        except Exception as e:
            logger.warning(f"AI question answering failed: {e}")
            answer = f"I'm sorry, I couldn't process your question at the moment. Error: {str(e)}"
        
        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "context_used": len(context) > 0,
            "message": "Question processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        rag = get_rag_system()
        if rag:
            try:
                stats = rag.get_stats()
            except Exception as e:
                stats = {"documents": 0, "message": f"RAG system error: {str(e)}"}
        else:
            stats = {"documents": 0, "message": "RAG system not available"}
        
        return {
            "status": "success",
            "system_stats": stats,
            "components": {
                "rag_system": rag is not None,
                "analyzer": get_analyzer() is not None,
                "file_processor": get_file_processor() is not None,
                "scraper": get_scraper() is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enhanced-upload")
async def enhanced_upload(
    file: UploadFile = File(...),
    provider: str = Form("gemini")
):
    """Enhanced file upload with comprehensive analysis and quiz generation"""
    try:
        logger.info(f"Enhanced processing file: {file.filename}")
        
        # Get enhanced processor
        processor = get_enhanced_processor()
        if not processor:
            raise HTTPException(status_code=500, detail="Enhanced processor not available")
        
        # Read file content
        content = await file.read()
        
        # Process file with enhanced processor
        result = processor.process_document(content, file.filename)
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Processing failed"))
        
        # Generate quiz if content is substantial
        quiz_result = None
        quiz_generator = get_quiz_generator()
        if quiz_generator and result.get("content"):
            try:
                quiz_result = quiz_generator.generate_quiz(
                    result["content"], 
                    result.get("type", "text"),
                    min_questions=10,
                    max_questions=20
                )
            except Exception as e:
                logger.warning(f"Quiz generation failed: {e}")
        
        # Get RAG system and analyzer for additional analysis
        rag = get_rag_system()
        analyzer = get_analyzer()
        
        additional_analysis = {}
        if rag and analyzer and result.get("content"):
            try:
                # Store in RAG system
                rag.add_document(result["content"], {"filename": file.filename, "type": "enhanced_upload"})
                
                # Generate AI analysis
                summary = await analyzer.enhanced_generate_recap(result["content"])
                concepts = await analyzer.enhanced_extract_key_concepts(result["content"])
                
                additional_analysis = {
                    "ai_summary": summary,
                    "ai_concepts": concepts
                }
            except Exception as e:
                logger.warning(f"AI analysis failed: {e}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "file_type": result.get("type", "unknown"),
            "analysis": result.get("analysis", {}),
            "quiz": quiz_result.get("quiz") if quiz_result and quiz_result.get("success") else None,
            "additional_analysis": additional_analysis,
            "message": "File processed with enhanced analysis and quiz generation"
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-quiz")
async def generate_quiz(
    content: str = Form(...),
    content_type: str = Form("text"),
    min_questions: int = Form(10),
    max_questions: int = Form(20),
    difficulty: str = Form("medium")
):
    """Generate quiz from content"""
    try:
        logger.info(f"Generating quiz for content type: {content_type}")
        
        quiz_generator = get_quiz_generator()
        if not quiz_generator:
            raise HTTPException(status_code=500, detail="Quiz generator not available")
        
        result = quiz_generator.generate_quiz(
            content, content_type, min_questions, max_questions, difficulty
        )
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Quiz generation failed"))
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating quiz: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start-conversation")
async def start_conversation(
    topic: str = Form(...),
    context: str = Form("{}")
):
    """Start a new learning conversation"""
    try:
        logger.info(f"Starting conversation about: {topic}")
        
        companion = get_learning_companion()
        if not companion:
            raise HTTPException(status_code=500, detail="Learning companion not available")
        
        # Parse context if provided
        try:
            context_dict = json.loads(context) if context else {}
        except:
            context_dict = {}
        
        result = companion.start_conversation(topic, context_dict)
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to start conversation"))
        
        return result
        
    except Exception as e:
        logger.error(f"Error starting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(
    message: str = Form(...),
    context: str = Form("{}")
):
    """Chat with the learning companion"""
    try:
        logger.info(f"Processing chat message: {message[:50]}...")
        
        companion = get_learning_companion()
        if not companion:
            raise HTTPException(status_code=500, detail="Learning companion not available")
        
        # Parse context if provided
        try:
            context_dict = json.loads(context) if context else {}
        except:
            context_dict = {}
        
        result = companion.respond_to_question(message, context_dict)
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to process message"))
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting Smart Content Agent Backend Server...")
    logger.info("Server will be available at: http://127.0.0.1:8000")
    logger.info("API documentation at: http://127.0.0.1:8000/docs")
    
    uvicorn.run(
        "backend_server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
