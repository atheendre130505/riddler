#!/usr/bin/env python3
"""
Simple test server to verify FastAPI is working
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Test Server")

@app.get("/")
async def root():
    return {"message": "Hello World! Server is working!"}

@app.get("/test", response_class=HTMLResponse)
async def test_html():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1>Test Server is Working!</h1>
        <p>If you can see this, the server is running correctly.</p>
    </body>
    </html>
    """

if __name__ == "__main__":
    print("Starting test server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

