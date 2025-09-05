# Smart Content Agent - Deployment Guide

## 🚀 Quick Deploy Options

### Option 1: Railway (Recommended - Free Tier Available)

1. **Connect to Railway:**
   - Go to [Railway.app](https://railway.app)
   - Sign up with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

2. **Configure Environment Variables:**
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   MISTRAL_API_KEY=your_mistral_api_key (optional)
   HUGGINGFACE_API_KEY=your_hf_api_key (optional)
   ```

3. **Deploy:**
   - Railway will automatically detect the `railway.json` config
   - Your app will be deployed at `https://your-app-name.railway.app`

### Option 2: Render (Free Tier Available)

1. **Connect to Render:**
   - Go to [Render.com](https://render.com)
   - Sign up with GitHub
   - Click "New" → "Web Service"
   - Connect your repository

2. **Configure Service:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Health Check Path: `/health`

3. **Add Environment Variables:**
   - Go to Environment tab
   - Add your API keys

### Option 3: Heroku (Paid)

1. **Install Heroku CLI:**
   ```bash
   # Ubuntu/Debian
   curl https://cli-assets.heroku.com/install.sh | sh
   
   # Or download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Deploy:**
   ```bash
   # Login to Heroku
   heroku login
   
   # Create app
   heroku create your-app-name
   
   # Set environment variables
   heroku config:set GOOGLE_API_KEY=your_key
   heroku config:set MISTRAL_API_KEY=your_key
   
   # Deploy
   git push heroku main
   ```

### Option 4: Docker Deployment

1. **Build Docker Image:**
   ```bash
   docker build -t smart-content-agent .
   ```

2. **Run Container:**
   ```bash
   docker run -p 8000:8000 \
     -e GOOGLE_API_KEY=your_key \
     -e MISTRAL_API_KEY=your_key \
     smart-content-agent
   ```

3. **Deploy to Cloud:**
   - Push to Docker Hub
   - Deploy to any cloud provider that supports Docker

## 🔧 Environment Variables

Create a `.env` file or set these in your cloud platform:

```bash
# Required for Gemini provider
GOOGLE_API_KEY=your_gemini_api_key

# Optional for Mistral provider
MISTRAL_API_KEY=your_mistral_api_key

# Optional for enhanced performance
HUGGINGFACE_API_KEY=your_hf_api_key

# Server configuration
PORT=8000
HOST=0.0.0.0
```

## 📊 Monitoring & Health Checks

- **Health Check Endpoint:** `/health`
- **API Documentation:** `/docs`
- **Main Application:** `/`

## 🛠️ Troubleshooting

### Common Issues:

1. **Port Issues:**
   - Make sure your app binds to `0.0.0.0` and uses `$PORT` environment variable

2. **Memory Issues:**
   - Some platforms have memory limits
   - Consider using lighter models or optimizing dependencies

3. **Build Failures:**
   - Check Python version compatibility
   - Ensure all dependencies are in `requirements.txt`

4. **API Key Issues:**
   - Verify environment variables are set correctly
   - Check API key permissions and quotas

## 🔒 Security Considerations

1. **Environment Variables:**
   - Never commit API keys to git
   - Use platform-specific secret management

2. **CORS Settings:**
   - Update CORS origins for production
   - Consider restricting to specific domains

3. **Rate Limiting:**
   - Implement rate limiting for production use
   - Monitor usage and costs

## 📈 Performance Optimization

1. **Caching:**
   - Enable Redis caching for production
   - Use CDN for static assets

2. **Database:**
   - Use persistent storage for RAG data
   - Consider database optimization

3. **Scaling:**
   - Use load balancers for high traffic
   - Implement horizontal scaling

## 🎯 Deployment Checklist

- [ ] Repository is public or has proper access
- [ ] All dependencies are in `requirements.txt`
- [ ] Environment variables are configured
- [ ] Health check endpoint is working
- [ ] CORS settings are appropriate
- [ ] API keys have proper permissions
- [ ] Monitoring is set up
- [ ] Backup strategy is in place

## 📞 Support

If you encounter issues:

1. Check the deployment logs
2. Verify environment variables
3. Test locally first
4. Check platform-specific documentation
5. Review the troubleshooting section above

## 🎉 Success!

Once deployed, your Smart Content Agent will be available at your platform's URL. You can:

- Upload and process files
- Analyze web content
- Ask questions about content
- Generate quizzes and summaries

Enjoy your deployed AI-powered content analysis platform!
