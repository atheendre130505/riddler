# Smart Content Agent - Local Hosting Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Or if you prefer using a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
```bash
# The startup script will create a .env template for you
python start_server.py

# Edit the .env file with your API keys
nano .env  # or use your preferred editor
```

### 3. Start the Server
```bash
# Option 1: Use the startup script (recommended)
python start_server.py

# Option 2: Direct uvicorn command
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the Application
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🔑 Required API Keys

### Google Gemini API Key (Recommended)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

### Mistral API Key (Optional)
1. Go to [Mistral Console](https://console.mistral.ai/)
2. Create an account and get your API key
3. Add it to your `.env` file:
   ```
   MISTRAL_API_KEY=your_api_key_here
   ```

### Hugging Face API Key (Optional)
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token
3. Add it to your `.env` file:
   ```
   HUGGINGFACE_API_KEY=your_api_key_here
   ```

## 📁 Project Structure

```
teach-ai/
├── main.py                 # FastAPI application
├── start_server.py         # Startup script
├── advanced_rag.py         # Advanced RAG system
├── enhanced_analyzer.py    # Enhanced content analyzer
├── processor.py            # File processor
├── scraper.py              # Web scraper
├── templates/
│   └── index.html         # Web interface
├── static/                # Static files
├── uploads/               # Temporary file uploads
├── advanced_rag/          # RAG data storage
├── hybrid_rag/            # Hybrid RAG data
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
└── README_LOCAL_HOSTING.md # This file
```

## 🎯 Features

### File Processing
- **Supported Formats**: PDF, DOCX, TXT
- **Drag & Drop**: Upload files via web interface
- **Content Extraction**: Automatic text extraction and processing

### URL Processing
- **Web Scraping**: Extract content from any URL
- **Content Analysis**: Automatic summarization and quiz generation

### AI-Powered Analysis
- **Multiple Providers**: Gemini, Mistral, Hugging Face
- **Smart Summaries**: Context-aware content summarization
- **Quiz Generation**: Automatic question and answer creation
- **Key Concepts**: Extract important topics and concepts

### Advanced RAG System
- **Vector Search**: ChromaDB and FAISS integration
- **Hybrid Search**: Combines vector, keyword, and semantic search
- **Caching**: Performance optimization with intelligent caching
- **Multi-modal**: Support for different content types

### Interactive Q&A
- **Learning Questions**: Ask questions about uploaded content
- **Context Awareness**: Uses RAG system for relevant answers
- **Source Attribution**: Shows sources used for answers
- **Confidence Scoring**: Indicates answer reliability

## 🔧 Configuration

### Environment Variables
```bash
# Required for Gemini provider
GOOGLE_API_KEY=your_gemini_api_key

# Required for Mistral provider  
MISTRAL_API_KEY=your_mistral_api_key

# Optional for enhanced performance
HUGGINGFACE_API_KEY=your_hf_api_key

# Cache settings
CACHE_TTL=3600  # Cache time-to-live in seconds

# Server settings
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

### Provider Selection
The application supports multiple AI providers:

1. **Gemini 1.5 Pro** (Recommended)
   - Multimodal capabilities
   - High-quality responses
   - Free tier available

2. **Mistral Large**
   - Most powerful model
   - Excellent for complex analysis
   - Requires API key

3. **Hugging Face**
   - Free options available
   - Good for basic tasks
   - Local model support

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# Check Python version (3.8+ required)
python --version
```

#### 2. API Key Issues
```bash
# Verify your .env file exists and has correct keys
cat .env

# Check if keys are properly formatted (no spaces, quotes, etc.)
```

#### 3. Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process or use a different port
uvicorn main:app --port 8001
```

#### 4. Memory Issues
```bash
# For large files, you might need more memory
# Consider using smaller chunk sizes in the code
```

### Performance Optimization

#### 1. Enable Caching
- The system automatically caches embeddings and search results
- Adjust `CACHE_TTL` in `.env` for your needs

#### 2. Use SSD Storage
- Store the application on SSD for better performance
- Especially important for vector databases

#### 3. Monitor Resources
- Check CPU and memory usage during processing
- Large files may require more resources

## 📊 API Endpoints

### Core Endpoints
- `POST /upload` - Upload and process files
- `POST /process-url` - Process web URLs
- `POST /ask-question` - Ask learning questions
- `GET /health` - Health check

### Advanced Endpoints
- `POST /enhanced/ask` - Enhanced Q&A with context
- `GET /enhanced/conversation/{session_id}` - Get conversation history
- `POST /enhanced/stream` - Stream analysis results
- `GET /enhanced/metrics` - Performance metrics
- `GET /enhanced/rag/stats` - RAG system statistics

### Utility Endpoints
- `GET /rag-stats` - RAG system status
- `GET /mcp-status` - MCP system status
- `POST /enhanced/clear-cache` - Clear performance caches

## 🔒 Security Notes

1. **API Keys**: Never commit API keys to version control
2. **File Uploads**: Files are temporarily stored and automatically deleted
3. **CORS**: Currently configured for development (allows all origins)
4. **Rate Limiting**: Consider implementing rate limiting for production

## 🚀 Production Deployment

For production deployment, consider:

1. **Environment Variables**: Use secure environment variable management
2. **HTTPS**: Enable SSL/TLS encryption
3. **Rate Limiting**: Implement request rate limiting
4. **Monitoring**: Add application monitoring and logging
5. **Database**: Use persistent database for production data
6. **Load Balancing**: Consider load balancing for high traffic

## 📞 Support

If you encounter issues:

1. Check the logs in the terminal
2. Verify all dependencies are installed
3. Ensure API keys are correctly configured
4. Check the health endpoint: http://localhost:8000/health

## 🎉 Enjoy!

Your Smart Content Agent is now running locally! Upload some content and start exploring the AI-powered analysis features.

