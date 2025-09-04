from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from datetime import datetime
from typing import Optional
import aiofiles

from processor import FileProcessor
from enhanced_analyzer import EnhancedContentAnalyzer
from scraper import WebScraper

app = FastAPI(title="Smart Content Agent", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
file_processor = FileProcessor()
scraper = WebScraper()

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page"""
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), provider: str = Form("gemini")):
    """
    Upload and process a file (PDF, DOCX, TXT)
    Returns content summary and generated quiz questions
    """
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1].lower()
        file_path = f"uploads/{file_id}{file_extension}"
        
        # Save uploaded file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Process the file
        result = file_processor.process_file(file_path)
        
        if not result or not result.get("content"):
            raise HTTPException(status_code=400, detail="Could not extract content from file")
        
        # Initialize enhanced analyzer with selected provider
        analyzer = EnhancedContentAnalyzer(provider=provider)
        
        # Create conversation session
        session_id = analyzer.create_conversation_session("default", {
            "file_id": file_id,
            "filename": file.filename,
            "provider": provider
        })
        
        # Generate enhanced recap and questions (async)
        import asyncio
        recap = await analyzer.enhanced_generate_recap(result["content"], "medium", session_id)
        questions = await analyzer.enhanced_create_questions(result["content"], 5, session_id)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "file_type": result.get("file_type", "unknown"),
            "content_length": len(result["content"]),
            "recap": recap,
            "quiz": questions,
            "key_concepts": await analyzer.enhanced_extract_key_concepts(result["content"], session_id),
            "session_id": session_id
        }
        
    except Exception as e:
        # Clean up file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/process-url")
async def process_url(url: str = Form(...), provider: str = Form("gemini")):
    """
    Process a web URL and extract content
    Returns content summary and generated quiz questions
    """
    try:
        # Scrape the URL
        data = scraper.scrape_url(url)
        
        if not data or not data.get("content"):
            raise HTTPException(status_code=400, detail="Could not extract content from URL")
        
        # Initialize analyzer with selected provider
        analyzer = ContentAnalyzer(provider=provider)
        
        # Generate recap and questions (async)
        import asyncio
        recap = await analyzer.generate_recap(data["content"])
        questions = await analyzer.create_questions(data["content"])
        
        return {
            "url": url,
            "title": data.get("title", ""),
            "content_length": len(data["content"]),
            "recap": recap,
            "quiz": questions,
            "key_concepts": await analyzer.extract_key_concepts(data["content"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

@app.get("/recap/{content_id}")
async def get_recap(content_id: str, length: str = "medium"):
    """
    Get content summary by ID (placeholder for future storage implementation)
    """
    return {"message": "Recap endpoint - content storage not yet implemented", "content_id": content_id, "length": length}

@app.get("/quiz/{content_id}")
async def get_quiz(content_id: str, count: int = 10):
    """
    Get quiz questions by ID (placeholder for future storage implementation)
    """
    return {"message": "Quiz endpoint - content storage not yet implemented", "content_id": content_id, "count": count}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Test analyzer with MCP and RAG
    test_analyzer = ContentAnalyzer(provider="gemini", enable_mcp=True, enable_rag=True)
    
    return {"status": "healthy", "components": {
        "file_processor": "ready",
        "analyzer": "ready", 
        "scraper": "ready",
        "mcp": "ready" if hasattr(test_analyzer, 'mcp_client') else "fallback",
        "rag": "ready" if hasattr(test_analyzer, 'rag_system') else "fallback"
    }}

@app.get("/rag-stats")
async def rag_stats():
    """Get RAG system statistics"""
    test_analyzer = ContentAnalyzer(provider="gemini", enable_mcp=True, enable_rag=True)
    if hasattr(test_analyzer, 'rag_system'):
        return test_analyzer.rag_system.get_document_stats()
    else:
        return {"error": "RAG system not available"}

@app.get("/mcp-status")
async def mcp_status():
    """Get MCP system status"""
    test_analyzer = ContentAnalyzer(provider="gemini", enable_mcp=True, enable_rag=True)
    if hasattr(test_analyzer, 'mcp_client'):
        return await test_analyzer.mcp_client.get_context_status()
    else:
        return {"error": "MCP client not available"}

@app.post("/ask-question")
async def ask_learning_question(
    question: str = Form(...),
    content_id: str = Form(None),
    context: str = Form(None),
    provider: str = Form("gemini")
):
    """
    Ask learning-focused questions with RAG + MCP + Gemini 2.5 Flash web search
    
    Args:
        question: The learning question to ask
        content_id: Optional content ID to focus on specific document
        context: Optional additional context
        provider: AI provider to use (default: gemini)
        
    Returns:
        Comprehensive answer with sources and learning insights
    """
    try:
        # Initialize analyzer with MCP and RAG
        analyzer = ContentAnalyzer(provider=provider, enable_mcp=True, enable_rag=True)
        
        # Get comprehensive answer using RAG + MCP + web search
        result = await analyzer.ask_learning_question(
            question=question,
            content_id=content_id,
            context=context
        )
        
        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "learning_insights": result["learning_insights"],
            "question_type": result["question_type"],
            "confidence": result["confidence"],
            "rag_context_used": result["rag_context_used"],
            "mcp_reasoning_used": result["mcp_reasoning_used"],
            "provider": provider,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing learning question: {str(e)}")
        return {
            "error": f"Failed to process question: {str(e)}",
            "question": question,
            "provider": provider
        }

# Enhanced endpoints with advanced features

@app.post("/enhanced/ask")
async def enhanced_ask_question(question: str = Form(...), session_id: str = Form(None), 
                               provider: str = Form("gemini")):
    """
    Enhanced learning question endpoint with conversation context
    """
    try:
        analyzer = EnhancedContentAnalyzer(provider=provider)
        
        # Use provided session or create new one
        if not session_id:
            session_id = analyzer.create_conversation_session("default")
        
        result = await analyzer.enhanced_ask_learning_question(
            question=question,
            session_id=session_id
        )
        
        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "learning_insights": result["learning_insights"],
            "question_type": result["question_type"],
            "confidence": result["confidence"],
            "rag_context_used": result["rag_context_used"],
            "conversation_context_used": result["conversation_context_used"],
            "session_id": session_id,
            "provider": provider,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": str(e), "question": question, "provider": provider}

@app.get("/enhanced/conversation/{session_id}")
async def get_conversation_history(session_id: str, limit: int = 10):
    """
    Get conversation history for a session
    """
    try:
        analyzer = EnhancedContentAnalyzer()
        history = analyzer.get_conversation_history(session_id, limit)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        return {"error": str(e), "session_id": session_id}

@app.post("/enhanced/stream")
async def stream_analysis(content: str = Form(...), analysis_type: str = Form("recap"), 
                         session_id: str = Form(None)):
    """
    Stream analysis results for real-time experience
    """
    try:
        analyzer = EnhancedContentAnalyzer()
        
        if not session_id:
            session_id = analyzer.create_conversation_session("default")
        
        # Create streaming response
        from fastapi.responses import StreamingResponse
        import json
        
        async def generate_stream():
            async for chunk in analyzer.stream_analysis(content, analysis_type, session_id):
                yield f"data: {json.dumps(chunk)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/enhanced/metrics")
async def get_performance_metrics():
    """
    Get comprehensive performance metrics
    """
    try:
        analyzer = EnhancedContentAnalyzer()
        metrics = analyzer.get_performance_metrics()
        return metrics
    except Exception as e:
        return {"error": str(e)}

@app.post("/enhanced/clear-cache")
async def clear_performance_cache():
    """
    Clear all performance caches
    """
    try:
        analyzer = EnhancedContentAnalyzer()
        analyzer.clear_performance_cache()
        return {"status": "success", "message": "All caches cleared"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/enhanced/rag/stats")
async def get_rag_stats():
    """
    Get RAG system statistics
    """
    try:
        analyzer = EnhancedContentAnalyzer()
        stats = analyzer.advanced_rag.get_stats()
        return stats
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
