#!/bin/bash

# Smart Content Agent - Quick Start Script
# This script sets up and starts the Smart Content Agent locally

echo "🎯 Smart Content Agent - Local Setup"
echo "===================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ Python and pip are available"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads static advanced_rag hybrid_rag

# Check if .env exists, if not create template
if [ ! -f ".env" ]; then
    echo "📝 Creating .env template..."
    cat > .env << EOF
# Smart Content Agent Environment Variables
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
EOF
    echo "⚠️  Please edit .env file and add your API keys before starting the server"
    echo "   You can edit it with: nano .env"
    echo ""
    read -p "Press Enter to continue after adding your API keys..."
fi

# Start the server
echo "🚀 Starting Smart Content Agent server..."
echo "📱 Server will be available at: http://localhost:8000"
echo "📚 API documentation at: http://localhost:8000/docs"
echo "🛑 Press Ctrl+C to stop the server"
echo "----------------------------------------"

# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

