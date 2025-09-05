"""
Simplified Firebase Functions for Smart Content Agent
"""

import os
import json
import logging
from functions_framework import http

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@http
def api(request):
    """Main Firebase Function entry point"""
    try:
        # Set CORS headers
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Content-Type': 'application/json'
        }
        
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return ('', 204, headers)
        
        # Get the path
        path = request.path
        
        # Route to appropriate handler
        if path == '/api/' or path == '/api':
            return handle_root(headers)
        elif path == '/api/health':
            return handle_health(headers)
        elif path == '/api/upload':
            return handle_upload(request, headers)
        elif path == '/api/process-url':
            return handle_process_url(request, headers)
        else:
            return handle_not_found(headers)
            
    except Exception as e:
        logger.error(f"Error in Firebase Function: {str(e)}")
        return (json.dumps({"error": str(e)}), 500, headers)

def handle_root(headers):
    """Handle root endpoint"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Smart Content Agent - Firebase</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                background: white;
                color: #333;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2d3748;
                text-align: center;
                margin-bottom: 30px;
            }
            .status {
                background: #c6f6d5;
                color: #22543d;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                text-align: center;
            }
            .endpoints {
                background: #f7fafc;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .endpoint {
                margin: 10px 0;
                padding: 10px;
                background: white;
                border-radius: 5px;
                border-left: 4px solid #667eea;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Smart Content Agent</h1>
            <div class="status">
                ✅ Successfully deployed on Firebase Functions!
            </div>
            <div class="endpoints">
                <h3>Available Endpoints:</h3>
                <div class="endpoint">
                    <strong>GET /api/health</strong> - Health check
                </div>
                <div class="endpoint">
                    <strong>POST /api/upload</strong> - Upload and process files
                </div>
                <div class="endpoint">
                    <strong>POST /api/process-url</strong> - Process web URLs
                </div>
            </div>
            <p style="text-align: center; color: #718096;">
                Your Smart Content Agent is running on Firebase Functions! 🚀
            </p>
        </div>
    </body>
    </html>
    """
    
    headers['Content-Type'] = 'text/html'
    return (html_content, 200, headers)

def handle_health(headers):
    """Handle health check"""
    response = {
        "status": "healthy",
        "message": "Smart Content Agent is running on Firebase Functions",
        "platform": "Firebase",
        "endpoints": [
            "/api/health",
            "/api/upload", 
            "/api/process-url"
        ]
    }
    return (json.dumps(response), 200, headers)

def handle_upload(request, headers):
    """Handle file upload"""
    try:
        # For now, return a simple response
        # In a full implementation, you'd process the file here
        response = {
            "message": "File upload endpoint is working",
            "status": "success",
            "note": "Full file processing will be implemented in the next version"
        }
        return (json.dumps(response), 200, headers)
    except Exception as e:
        return (json.dumps({"error": str(e)}), 500, headers)

def handle_process_url(request, headers):
    """Handle URL processing"""
    try:
        # For now, return a simple response
        # In a full implementation, you'd process the URL here
        response = {
            "message": "URL processing endpoint is working",
            "status": "success",
            "note": "Full URL processing will be implemented in the next version"
        }
        return (json.dumps(response), 200, headers)
    except Exception as e:
        return (json.dumps({"error": str(e)}), 500, headers)

def handle_not_found(headers):
    """Handle 404 errors"""
    response = {
        "error": "Endpoint not found",
        "available_endpoints": [
            "/api/",
            "/api/health",
            "/api/upload",
            "/api/process-url"
        ]
    }
    return (json.dumps(response), 404, headers)
