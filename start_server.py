#!/usr/bin/env python3
"""
Smart Content Agent - Local Server Startup Script
This script starts the FastAPI server with all necessary configurations.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import aiofiles
        import requests
        import beautifulsoup4
        import pypdf2
        import python_docx
        import google_generativeai
        import mistralai
        import sklearn
        import numpy
        import aiohttp
        logger.info("✅ All required packages are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing required package: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False

def check_environment():
    """Check environment variables and create .env if needed"""
    env_file = Path(".env")
    
    if not env_file.exists():
        logger.info("📝 Creating .env file template...")
        env_content = """# Smart Content Agent Environment Variables
# Copy this file and add your API keys

# Google Gemini API Key (Required for Gemini provider)
# Get it from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_gemini_api_key_here

# Mistral API Key (Required for Mistral provider)
# Get it from: https://console.mistral.ai/
MISTRAL_API_KEY=your_mistral_api_key_here

# Hugging Face API Key (Optional, for better performance)
# Get it from: https://huggingface.co/settings/tokens
HUGGINGFACE_API_KEY=your_hf_api_key_here

# Cache TTL in seconds (default: 3600 = 1 hour)
CACHE_TTL=3600

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        logger.info("✅ Created .env file template")
        logger.warning("⚠️  Please edit .env file and add your API keys")
    else:
        logger.info("✅ .env file exists")

def create_directories():
    """Create necessary directories"""
    directories = ["uploads", "static", "advanced_rag", "hybrid_rag"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Directory '{directory}' ready")

def start_server():
    """Start the FastAPI server"""
    try:
        logger.info("🚀 Starting Smart Content Agent server...")
        logger.info("📱 Server will be available at: http://localhost:8000")
        logger.info("📚 API documentation at: http://localhost:8000/docs")
        logger.info("🛑 Press Ctrl+C to stop the server")
        logger.info("-" * 50)
        
        # Start the server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload",
            "--log-level", "info"
        ])
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting server: {e}")

def main():
    """Main function"""
    logger.info("🎯 Smart Content Agent - Local Server Setup")
    logger.info("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    check_environment()
    
    # Create directories
    create_directories()
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()

