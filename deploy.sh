#!/bin/bash

# Smart Content Agent - Quick Deploy Script
echo "🚀 Smart Content Agent - Deployment Options"
echo "=========================================="

echo ""
echo "Your code has been pushed to GitHub: https://github.com/atheendre130505/riddler"
echo ""

echo "Choose your deployment platform:"
echo ""
echo "1. Railway (Recommended - Free tier available)"
echo "   - Go to: https://railway.app"
echo "   - Sign up with GitHub"
echo "   - Click 'New Project' → 'Deploy from GitHub repo'"
echo "   - Select your repository"
echo "   - Add environment variables:"
echo "     GOOGLE_API_KEY=your_gemini_api_key"
echo "     MISTRAL_API_KEY=your_mistral_api_key (optional)"
echo "     HUGGINGFACE_API_KEY=your_hf_api_key (optional)"
echo ""

echo "2. Render (Free tier available)"
echo "   - Go to: https://render.com"
echo "   - Sign up with GitHub"
echo "   - Click 'New' → 'Web Service'"
echo "   - Connect your repository"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: uvicorn main:app --host 0.0.0.0 --port \$PORT"
echo "   - Health Check Path: /health"
echo ""

echo "3. Heroku (Paid)"
echo "   - Install Heroku CLI"
echo "   - Run: heroku create your-app-name"
echo "   - Run: heroku config:set GOOGLE_API_KEY=your_key"
echo "   - Run: git push heroku main"
echo ""

echo "4. Docker (Any cloud provider)"
echo "   - Build: docker build -t smart-content-agent ."
echo "   - Run: docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key smart-content-agent"
echo ""

echo "📚 For detailed instructions, see DEPLOYMENT.md"
echo ""

echo "🔧 Required Environment Variables:"
echo "   GOOGLE_API_KEY (required for Gemini)"
echo "   MISTRAL_API_KEY (optional for Mistral)"
echo "   HUGGINGFACE_API_KEY (optional for enhanced performance)"
echo ""

echo "✅ Your app is ready to deploy!"
echo "   Repository: https://github.com/atheendre130505/riddler"
echo "   Local testing: http://127.0.0.1:8000"