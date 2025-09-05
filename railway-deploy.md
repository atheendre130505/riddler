# 🚀 One-Click Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/deploy?repository=https://github.com/atheendre130505/riddler)

## Quick Setup

1. **Click the button above** to deploy to Railway
2. **Sign up/Login** with GitHub
3. **Add Environment Variables:**
   - `GOOGLE_API_KEY` - Your Gemini API key (required)
   - `MISTRAL_API_KEY` - Your Mistral API key (optional)
   - `HUGGINGFACE_API_KEY` - Your Hugging Face API key (optional)

4. **Deploy!** Railway will automatically:
   - Clone your repository
   - Install dependencies
   - Start your application
   - Provide a public URL

## After Deployment

Your Smart Content Agent will be available at:
`https://your-app-name.railway.app`

## Features Available

- ✅ File upload and processing (PDF, DOCX, TXT)
- ✅ URL content extraction and analysis
- ✅ AI-powered content summarization
- ✅ Interactive Q&A system
- ✅ Quiz generation
- ✅ Key concept extraction
- ✅ Multiple AI providers (Gemini, Mistral, Hugging Face)

## API Endpoints

- **Main App:** `/`
- **API Docs:** `/docs`
- **Health Check:** `/health`
- **File Upload:** `POST /upload`
- **URL Processing:** `POST /process-url`

## Need Help?

- Check the [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed instructions
- Review the [README_LOCAL_HOSTING.md](./README_LOCAL_HOSTING.md) for local setup
- Check Railway's [documentation](https://docs.railway.app/)

---

**Your Smart Content Agent is ready to transform any content into interactive learning experiences!** 🎉
