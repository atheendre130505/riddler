# 🧠 Smart Content Agent

**AI-Powered Content Analysis & Learning Platform**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20Now-green?style=for-the-badge)](https://atheendre130505.github.io/riddler/)
[![GitHub](https://img.shields.io/badge/GitHub-View%20Source-blue?style=for-the-badge)](https://github.com/atheendre130505/riddler)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=for-the-badge)](https://fastapi.tiangolo.com)

## 🌐 **Live Webapp**

**👉 [Open the Smart Content Agent Webapp](https://atheendre130505.github.io/riddler/)**

## 🚀 Features

### 📁 **Document Processing**
- Upload PDF, DOCX, and TXT files
- AI-powered content extraction
- Automatic text cleaning and normalization

### 🔍 **Advanced RAG System**
- **Vector Search**: ChromaDB for semantic similarity
- **Keyword Search**: TF-IDF for exact matches
- **Hybrid Search**: Combines multiple search methods
- **Caching**: LRU cache for performance optimization

### 🤖 **AI-Powered Analysis**
- **Multi-Provider Support**: Gemini, Mistral, Hugging Face
- **Content Summarization**: Generate intelligent summaries
- **Key Concept Extraction**: Identify important topics
- **Quiz Generation**: Create educational questions
- **Interactive Q&A**: Context-aware question answering

### 🌐 **Web Content Processing**
- URL content extraction
- BeautifulSoup for HTML parsing
- Content cleaning and formatting

## 🏗️ Architecture

```
Smart Content Agent
├── Frontend (GitHub Pages)
│   └── public/index.html
├── Backend API (FastAPI)
│   ├── backend_server.py
│   ├── advanced_rag.py
│   ├── enhanced_analyzer.py
│   ├── processor.py
│   └── scraper.py
└── AI Integration
    ├── Google Gemini
    ├── Mistral AI
    └── Hugging Face
```

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/atheendre130505/riddler.git
   cd riddler
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the backend server**
   ```bash
   python backend_server.py
   ```

4. **Open the frontend**
   - Open `public/index.html` in your browser
   - Or visit the [live demo](https://atheendre130505.github.io/riddler/)

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key
MISTRAL_API_KEY=your_mistral_api_key
HUGGINGFACE_API_KEY=your_hf_api_key
```

### API Keys

- **Google Gemini**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Mistral AI**: Get your API key from [Mistral AI Console](https://console.mistral.ai/)
- **Hugging Face**: Get your API key from [Hugging Face Settings](https://huggingface.co/settings/tokens)

## 📚 Usage

### 1. **Upload Documents**
- Click "Upload Files" tab
- Drag and drop or select PDF, DOCX, or TXT files
- Click "Analyze Content" to process

### 2. **Process URLs**
- Click "Process URL" tab
- Enter a web URL
- Click "Process URL" to extract content

### 3. **Ask Questions**
- Click "Ask Questions" tab
- Type your question about the content
- Get AI-powered answers

## 🔬 Technical Details

### **Advanced RAG System**
```python
class AdvancedRAG:
    - ChromaDB: Vector database for embeddings
    - FAISS: Fast similarity search
    - Sentence Transformers: Text embeddings
    - Hybrid Search: Multiple search strategies
    - Caching: Performance optimization
```

### **AI Integration**
```python
class EnhancedContentAnalyzer:
    - Multi-provider AI support
    - Content summarization
    - Key concept extraction
    - Quiz generation
    - Interactive Q&A
```

### **Document Processing**
```python
class FileProcessor:
    - PDF: PyPDF2 extraction
    - DOCX: python-docx processing
    - TXT: Direct text handling
    - Content cleaning and normalization
```

## 🌐 API Endpoints

### Backend API (FastAPI)

- `GET /` - API documentation and status
- `GET /health` - Health check
- `GET /stats` - System statistics
- `POST /upload` - File upload and processing
- `POST /process-url` - URL content extraction
- `POST /ask-question` - Interactive Q&A

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Upload file
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf" \
  -F "provider=gemini"

# Process URL
curl -X POST http://localhost:8000/process-url \
  -F "url=https://example.com" \
  -F "provider=gemini"

# Ask question
curl -X POST http://localhost:8000/ask-question \
  -F "question=What is the main topic?" \
  -F "session_id=default"
```

## 🚀 Deployment

### **Frontend (GitHub Pages)**
The frontend is automatically deployed to GitHub Pages:
- **URL**: https://atheendre130505.github.io/riddler/
- **Source**: `public/index.html`

### **Backend (Railway/Render/Heroku)**
Deploy the backend to any Python hosting service:

1. **Railway**
   ```bash
   railway login
   railway up
   ```

2. **Render**
   - Connect GitHub repository
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `python backend_server.py`

3. **Heroku**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## 📊 Performance

- **Lazy Loading**: Components initialize only when needed
- **Caching**: LRU cache for frequently accessed data
- **Async Processing**: Non-blocking I/O operations
- **Connection Pooling**: Efficient database connections
- **Error Recovery**: Graceful degradation on failures

## 🔒 Security

- **Input Validation**: Sanitize all user inputs
- **File Type Checking**: Validate uploaded files
- **CORS Configuration**: Secure cross-origin requests
- **Error Sanitization**: Don't expose sensitive information
- **Rate Limiting**: Prevent abuse

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [Google Gemini](https://ai.google.dev/) - AI provider
- [Mistral AI](https://mistral.ai/) - AI provider
- [Hugging Face](https://huggingface.co/) - AI models

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/atheendre130505/riddler/issues)
- **Discussions**: [GitHub Discussions](https://github.com/atheendre130505/riddler/discussions)
- **Email**: [Contact](mailto:your-email@example.com)

---

**Built with ❤️ by [Your Name](https://github.com/atheendre130505)**

**🌟 Star this repository if you found it helpful!**