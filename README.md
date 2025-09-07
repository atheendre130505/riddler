# Simple Content Analyzer

A minimal AI-powered content analysis tool that uses Google Gemini to analyze uploaded files and answer questions about them.

## Features

- Upload PDF, DOCX, or TXT files
- Get AI-generated summaries using Google Gemini
- Ask questions about the uploaded content
- Simple, clean interface

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Google API key:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

3. Start the server:
```bash
python server.py
```

4. Open `index.html` in your browser

## Usage

1. Upload a file (PDF, DOCX, or TXT)
2. Wait for the AI analysis and summary
3. Ask questions about the content

## API Endpoints

- `POST /upload` - Upload and analyze a file
- `POST /ask` - Ask a question about uploaded content
- `GET /` - Health check

That's it! Simple and working.
