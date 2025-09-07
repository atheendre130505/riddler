import os
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
import PyPDF2
from docx import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Simple Content Analyzer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.warning("GOOGLE_API_KEY not found. Please set it as an environment variable.")
    model = None
else:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')

# Store uploaded content in memory (simple approach)
uploaded_content = ""

def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from various file formats"""
    try:
        if filename.endswith('.pdf'):
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        
        elif filename.endswith(('.docx', '.doc')):
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        
        elif filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Simple Content Analyzer API", "status": "running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and analyze a file"""
    global uploaded_content
    
    try:
        logger.info(f"Processing file: {file.filename}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extract text
            extracted_text = extract_text_from_file(temp_file_path, file.filename)
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                raise HTTPException(status_code=400, detail="No meaningful content found in file")
            
            # Store content for Q&A
            uploaded_content = extracted_text
            
            # Generate summary using Gemini
            if model:
                summary_prompt = f"""
                Please provide a clear and concise summary of the following content:
                
                {extracted_text}
                
                The summary should be:
                - 2-3 paragraphs long
                - Capture the main points and key information
                - Be easy to understand
                """
                
                summary_response = model.generate_content(summary_prompt)
                summary = summary_response.text if summary_response.text else "Unable to generate summary"
            else:
                # Fallback summary when API key is not available
                words = extracted_text.split()
                summary = " ".join(words[:50]) + "..." if len(words) > 50 else extracted_text
                summary = f"Content Summary (API key not configured): {summary}"
            
            return {
                "status": "success",
                "filename": file.filename,
                "content_length": len(extracted_text),
                "summary": summary,
                "message": "File processed successfully"
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """Ask a question about the uploaded content"""
    global uploaded_content
    
    if not uploaded_content:
        raise HTTPException(status_code=400, detail="No content uploaded. Please upload a file first.")
    
    try:
        logger.info(f"Processing question: {question}")
        
        # Generate answer using Gemini
        if model:
            answer_prompt = f"""
            Based on the following content, please answer this question: {question}
            
            Content:
            {uploaded_content}
            
            Please provide a helpful and accurate answer based on the content. If the question cannot be answered from the content, please say so.
            """
            
            answer_response = model.generate_content(answer_prompt)
            answer = answer_response.text if answer_response.text else "Unable to generate answer"
        else:
            # Fallback answer when API key is not available
            answer = f"Question: {question}\n\nAnswer: I cannot provide AI-generated answers because the Google API key is not configured. Please set the GOOGLE_API_KEY environment variable to enable AI features."
        
        return {
            "status": "success",
            "question": question,
            "answer": answer
        }
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
