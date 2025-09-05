from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from datetime import datetime
from typing import Optional
import aiofiles
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Content Agent", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page"""
    try:
        with open("templates/index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        logger.error(f"Error loading template: {e}")
        return HTMLResponse(content="<h1>Error loading page</h1><p>Please check the server logs.</p>", status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Server is running"}

@app.get("/test")
async def test():
    """Test endpoint"""
    return {"message": "Test endpoint working"}

# Lazy loading of components
_components_loaded = False
_file_processor = None
_scraper = None

def get_components():
    global _components_loaded, _file_processor, _scraper
    
    if not _components_loaded:
        try:
            from processor import FileProcessor
            from scraper import WebScraper
            
            _file_processor = FileProcessor()
            _scraper = WebScraper()
            _components_loaded = True
            logger.info("Components loaded successfully")
        except Exception as e:
            logger.error(f"Error loading components: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading components: {str(e)}")
    
    return _file_processor, _scraper

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), provider: str = Form("gemini")):
    """
    Upload and process a file (PDF, DOCX, TXT)
    Returns content summary and generated quiz questions
    """
    try:
        file_processor, _ = get_components()
        
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
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "file_type": result.get("file_type", "unknown"),
            "content_length": len(result["content"]),
            "content_preview": result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"],
            "message": "File processed successfully (analysis features coming soon)"
        }
        
    except Exception as e:
        # Clean up file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/process-url")
async def process_url(url: str = Form(...), provider: str = Form("gemini")):
    """
    Process a web URL and extract content
    Returns content summary and generated quiz questions
    """
    try:
        _, scraper = get_components()
        
        # Scrape the URL
        data = scraper.scrape_url(url)
        
        if not data or not data.get("content"):
            raise HTTPException(status_code=400, detail="Could not extract content from URL")
        
        return {
            "url": url,
            "title": data.get("title", ""),
            "content_length": len(data["content"]),
            "content_preview": data["content"][:500] + "..." if len(data["content"]) > 500 else data["content"],
            "message": "URL processed successfully (analysis features coming soon)"
        }
        
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Smart Content Agent server...")
    print("Server will be available at: http://127.0.0.1:8000")
    print("API documentation at: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

