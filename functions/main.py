"""
Firebase Functions entry point for Smart Content Agent
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from functions_framework import http
from main import app

@http
def api(request):
    """Firebase Function entry point for FastAPI app"""
    # Set environment variables for Firebase
    os.environ.setdefault('HOST', '0.0.0.0')
    os.environ.setdefault('PORT', '8080')
    
    # Import and run the FastAPI app
    from fastapi import Request
    from fastapi.responses import Response
    import asyncio
    
    # Handle the request
    return asyncio.run(handle_request(request))

async def handle_request(request):
    """Handle the incoming request"""
    from fastapi import Request
    from fastapi.responses import Response
    import json
    
    # Create a FastAPI request object
    body = request.get_data()
    headers = dict(request.headers)
    method = request.method
    url = request.url
    
    # Create a mock request object for FastAPI
    class MockRequest:
        def __init__(self, method, url, headers, body):
            self.method = method
            self.url = url
            self.headers = headers
            self._body = body
            
        async def body(self):
            return self._body
            
        async def form(self):
            return {}
            
        async def json(self):
            try:
                return json.loads(self._body.decode())
            except:
                return {}
    
    mock_request = MockRequest(method, url, headers, body)
    
    # Route the request to the appropriate FastAPI endpoint
    if request.path.startswith('/api/'):
        # Remove /api prefix and route to FastAPI
        path = request.path[4:]  # Remove '/api'
        
        # Handle different endpoints
        if path == '/' or path == '':
            return await handle_root()
        elif path == '/health':
            return await handle_health()
        elif path == '/upload' and method == 'POST':
            return await handle_upload(request)
        elif path == '/process-url' and method == 'POST':
            return await handle_process_url(request)
        else:
            return Response(
                content=json.dumps({"error": "Endpoint not found"}),
                status_code=404,
                headers={"Content-Type": "application/json"}
            )
    else:
        return await handle_root()

async def handle_root():
    """Handle root endpoint - serve the HTML page"""
    try:
        with open("templates/index.html", "r") as f:
            html_content = f.read()
        return Response(
            content=html_content,
            status_code=200,
            headers={"Content-Type": "text/html"}
        )
    except Exception as e:
        return Response(
            content=f"<h1>Error loading page</h1><p>{str(e)}</p>",
            status_code=500,
            headers={"Content-Type": "text/html"}
        )

async def handle_health():
    """Handle health check endpoint"""
    return Response(
        content='{"status": "healthy", "message": "Firebase Functions server is running"}',
        status_code=200,
        headers={"Content-Type": "application/json"}
    )

async def handle_upload(request):
    """Handle file upload endpoint"""
    try:
        # Import the upload handler from main
        from main import upload_file
        
        # Create a mock file object
        class MockFile:
            def __init__(self, filename, content):
                self.filename = filename
                self._content = content
                
            async def read(self):
                return self._content
        
        # Parse multipart form data
        content_type = request.headers.get('content-type', '')
        if 'multipart/form-data' in content_type:
            # Simple multipart parsing (you might want to use a proper library)
            body = request.get_data()
            # This is a simplified version - in production, use proper multipart parsing
            filename = "uploaded_file.txt"
            file_content = body
            
            mock_file = MockFile(filename, file_content)
            provider = "gemini"  # Default provider
            
            # Call the upload function
            result = await upload_file(mock_file, provider)
            
            return Response(
                content=json.dumps(result),
                status_code=200,
                headers={"Content-Type": "application/json"}
            )
        else:
            return Response(
                content=json.dumps({"error": "Invalid content type"}),
                status_code=400,
                headers={"Content-Type": "application/json"}
            )
    except Exception as e:
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            headers={"Content-Type": "application/json"}
        )

async def handle_process_url(request):
    """Handle URL processing endpoint"""
    try:
        # Parse JSON body
        body = request.get_data()
        data = json.loads(body.decode())
        
        url = data.get('url', '')
        provider = data.get('provider', 'gemini')
        
        # Import the URL processor from main
        from main import process_url
        
        # Call the process_url function
        result = await process_url(url, provider)
        
        return Response(
            content=json.dumps(result),
            status_code=200,
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            headers={"Content-Type": "application/json"}
        )
