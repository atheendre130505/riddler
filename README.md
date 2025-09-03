# 🧠 Smart Content Agent - RAG + MCP Powered Learning Platform

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced AI-powered learning platform that combines **RAG (Retrieval-Augmented Generation)** and **MCP (Model Context Protocol)** to create an intelligent content analysis and interactive Q&A system powered by the most powerful free multimodal LLM models.

## 🚀 Features

### 🧠 **Powerful AI Models**
- **Gemini 1.5 Pro** - Google's most powerful multimodal model
- **Mistral Large** - Most powerful free model with state-of-the-art performance
- **Hugging Face** - Open-source alternative with community models

### 📄 **File Processing**
- Upload and analyze **PDF, DOCX, and TXT** files
- Extract clean text content with metadata
- Support for multiple file formats

### 🌐 **Web Scraping**
- Process any URL to extract main content
- Clean HTML and remove navigation/ads
- Extract titles, content, and metadata

### 🎯 **Content Analysis**
- Generate comprehensive summaries
- Create quiz questions (multiple choice, true/false, short answer)
- Extract key concepts and topics
- All powered by advanced AI models

### 💬 **Interactive Learning Q&A**
- Ask questions about uploaded content
- Context-aware answers using RAG system
- MCP reasoning for enhanced responses
- Confidence scoring and source attribution
- Learning insights and connections

### 🔍 **Advanced RAG System**
- **168+ document chunks** stored in ChromaDB
- Hugging Face embeddings for semantic search
- Vector similarity search across knowledge base
- Persistent storage with fallback systems

### 🧩 **MCP Integration**
- Model Context Protocol for advanced reasoning
- Context management and session handling
- Enhanced analysis capabilities

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/atheendre130505/riddler.git
cd riddler
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
export GOOGLE_API_KEY="your_gemini_api_key"
export HUGGINGFACE_API_KEY="your_huggingface_api_key"
export MISTRAL_API_KEY="your_mistral_api_key"  # Optional
```

5. **Run the application**
```bash
python main.py
```

6. **Access the web interface**
Open your browser and go to: `http://localhost:8000`

## 🔑 API Keys Setup

### Google Gemini API
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a new project
3. Generate an API key
4. Set as `GOOGLE_API_KEY` environment variable

### Hugging Face API
1. Go to [Hugging Face](https://huggingface.co/)
2. Create an account and go to Settings
3. Generate an access token
4. Set as `HUGGINGFACE_API_KEY` environment variable

### Mistral AI API (Optional)
1. Go to [Mistral AI](https://console.mistral.ai/)
2. Create an account
3. Generate an API key
4. Set as `MISTRAL_API_KEY` environment variable

## 📖 Usage

### Web Interface
1. **Upload Files**: Drag and drop PDF, DOCX, or TXT files
2. **Process URLs**: Paste any URL for content analysis
3. **Ask Questions**: Use the interactive Q&A section after processing content
4. **Choose AI Provider**: Select between Gemini 1.5 Pro, Mistral Large, or Hugging Face

### API Endpoints

#### Upload File
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "provider=gemini"
```

#### Process URL
```bash
curl -X POST "http://localhost:8000/process-url" \
  -F "url=https://example.com" \
  -F "provider=gemini"
```

#### Ask Learning Question
```bash
curl -X POST "http://localhost:8000/ask-question" \
  -F "question=What is machine learning?" \
  -F "provider=gemini"
```

#### Health Check
```bash
curl "http://localhost:8000/health"
```

#### RAG Statistics
```bash
curl "http://localhost:8000/rag-stats"
```

## 🏗️ Architecture

```
Smart Content Agent/
├── main.py                 # FastAPI server and endpoints
├── analyzer.py             # Content analysis with AI providers
├── processor.py            # File processing utilities
├── scraper.py              # Web scraping functionality
├── enhanced_mcp.py         # Model Context Protocol implementation
├── hybrid_rag.py           # RAG system with ChromaDB
├── templates/
│   └── index.html          # Modern web interface
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### AI Providers
- **Gemini 1.5 Pro**: Most powerful multimodal model
- **Mistral Large**: State-of-the-art performance
- **Hugging Face**: Open-source alternative

### RAG System
- **ChromaDB**: Vector database for document storage
- **Hugging Face Embeddings**: Semantic search capabilities
- **TF-IDF Fallback**: Text-based similarity search

### MCP System
- **Enhanced Context Management**: Advanced reasoning capabilities
- **Session Handling**: Persistent context across interactions
- **Fallback Support**: Graceful degradation when MCP unavailable

## 📊 System Status

The system provides real-time status information:

```json
{
  "status": "healthy",
  "components": {
    "file_processor": "ready",
    "analyzer": "ready",
    "scraper": "ready",
    "mcp": "ready",
    "rag": "ready"
  }
}
```

## 🚀 Deployment

### Local Development
```bash
python main.py
```

### Production Deployment
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google** for Gemini 1.5 Pro API
- **Mistral AI** for Mistral Large model
- **Hugging Face** for open-source models and embeddings
- **ChromaDB** for vector database capabilities
- **FastAPI** for the excellent web framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/atheendre130505/riddler/issues) page
2. Create a new issue with detailed information
3. Join our community discussions

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=atheendre130505/riddler&type=Date)](https://star-history.com/#atheendre130505/riddler&Date)

---

**Built with ❤️ using the most powerful free AI models available**