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
            "stats": "/stats"
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
        
        # Process file based on type
        if file.filename.endswith('.pdf'):
            processed_content = processor.process_pdf(content)
        elif file.filename.endswith('.docx'):
            processed_content = processor.process_docx(content)
        elif file.filename.endswith('.txt'):
            processed_content = processor.process_txt(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
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
        rag.add_document(processed_content, {"filename": file.filename, "type": "upload"})
        
        # Generate analysis
        try:
            summary = await analyzer.enhanced_generate_recap(processed_content)
            concepts = await analyzer.enhanced_extract_key_concepts(processed_content)
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            summary = "AI analysis not available"
            concepts = ["Content processed successfully"]
        
        return {
            "status": "success",
            "filename": file.filename,
            "content_length": len(processed_content),
            "summary": summary,
            "key_concepts": concepts,
            "message": "File processed and analyzed successfully"
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
        
        # Search for relevant content
        try:
            search_results = rag.hybrid_search(question, n_results=3)
            context = "\n".join([doc["content"] for doc in search_results])
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            context = "No relevant content found"
        
        # Generate answer
        try:
            answer = await analyzer.enhanced_ask_learning_question(question, context, session_id)
        except Exception as e:
            logger.warning(f"AI question answering failed: {e}")
            answer = "I'm sorry, I couldn't process your question at the moment. Please try again later."
        
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
