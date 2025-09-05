# 🔥 Firebase Deployment Guide

## Smart Content Agent on Firebase

Deploy your Smart Content Agent to Firebase Functions and Hosting for a scalable, serverless solution.

## 🚀 Quick Deploy

### Prerequisites

1. **Install Firebase CLI:**
   ```bash
   npm install -g firebase-tools
   ```

2. **Login to Firebase:**
   ```bash
   firebase login
   ```

3. **Initialize Firebase in your project:**
   ```bash
   firebase init
   ```
   - Select "Functions" and "Hosting"
   - Choose your Firebase project
   - Use Python for Functions
   - Use `public` as public directory

### Deploy to Firebase

1. **Deploy Functions and Hosting:**
   ```bash
   firebase deploy
   ```

2. **Deploy only Functions:**
   ```bash
   firebase deploy --only functions
   ```

3. **Deploy only Hosting:**
   ```bash
   firebase deploy --only hosting
   ```

## 📁 Project Structure

```
teach-ai/
├── functions/
│   ├── main.py              # Firebase Functions entry point
│   ├── simple_main.py       # Simplified Functions version
│   └── requirements.txt     # Python dependencies
├── public/
│   └── index.html          # Static frontend
├── firebase.json           # Firebase configuration
├── .firebaserc            # Firebase project settings
├── firestore.rules        # Firestore security rules
└── firestore.indexes.json # Firestore indexes
```

## 🔧 Configuration

### Firebase Configuration (`firebase.json`)

- **Hosting:** Serves static files from `public/` directory
- **Functions:** Python 3.12 runtime
- **Rewrites:** Routes `/api/*` to Functions
- **CORS:** Configured for cross-origin requests

### Environment Variables

Set environment variables in Firebase Console:

1. Go to [Firebase Console](https://console.firebase.google.com)
2. Select your project
3. Go to Functions → Configuration
4. Add environment variables:

```bash
GOOGLE_API_KEY=your_gemini_api_key
MISTRAL_API_KEY=your_mistral_api_key
HUGGINGFACE_API_KEY=your_hf_api_key
```

## 🌐 API Endpoints

Your deployed app will have these endpoints:

- **Main App:** `https://your-project.web.app/`
- **Health Check:** `https://your-project.web.app/api/health`
- **File Upload:** `https://your-project.web.app/api/upload`
- **URL Processing:** `https://your-project.web.app/api/process-url`

## 🛠️ Development

### Local Development

1. **Start Firebase Emulator:**
   ```bash
   firebase emulators:start
   ```

2. **Test locally:**
   - Functions: `http://localhost:5001`
   - Hosting: `http://localhost:5000`

### Testing Functions

```bash
# Test health endpoint
curl https://your-project.web.app/api/health

# Test upload endpoint
curl -X POST https://your-project.web.app/api/upload \
  -H "Content-Type: application/json" \
  -d '{"test": true}'
```

## 📊 Monitoring

### Firebase Console

- **Functions:** Monitor execution, logs, and performance
- **Hosting:** View analytics and usage
- **Firestore:** Database management and security

### Logs

```bash
# View function logs
firebase functions:log

# View specific function logs
firebase functions:log --only api
```

## 🔒 Security

### Firestore Rules

The `firestore.rules` file defines:
- Authenticated users can read/write all documents
- Public documents are readable by everyone
- User-specific documents are protected

### CORS Configuration

CORS is configured in `firebase.json` to allow:
- All origins (`*`)
- Common HTTP methods
- Standard headers

## 🚀 Advanced Features

### Custom Domain

1. Go to Firebase Console → Hosting
2. Click "Add custom domain"
3. Follow the verification steps

### CDN

Firebase Hosting automatically provides:
- Global CDN
- HTTPS by default
- Automatic compression

### Scaling

Firebase Functions automatically:
- Scale based on demand
- Handle cold starts
- Manage resources

## 🐛 Troubleshooting

### Common Issues

1. **Function Timeout:**
   - Increase timeout in `firebase.json`
   - Optimize function code

2. **Memory Issues:**
   - Increase memory allocation
   - Optimize dependencies

3. **CORS Errors:**
   - Check CORS configuration
   - Verify headers

### Debug Mode

```bash
# Run with debug logging
firebase functions:shell

# Test functions locally
firebase emulators:start --debug
```

## 📈 Performance

### Optimization Tips

1. **Cold Starts:**
   - Use connection pooling
   - Minimize imports
   - Cache frequently used data

2. **Memory Usage:**
   - Monitor function memory
   - Optimize data structures
   - Use streaming for large files

3. **Response Time:**
   - Implement caching
   - Use async operations
   - Optimize database queries

## 🎯 Deployment Checklist

- [ ] Firebase CLI installed and authenticated
- [ ] Project initialized with Functions and Hosting
- [ ] Environment variables configured
- [ ] Functions code tested locally
- [ ] Firestore rules configured
- [ ] CORS settings appropriate
- [ ] Custom domain configured (optional)
- [ ] Monitoring set up

## 📞 Support

### Firebase Resources

- [Firebase Documentation](https://firebase.google.com/docs)
- [Functions Documentation](https://firebase.google.com/docs/functions)
- [Hosting Documentation](https://firebase.google.com/docs/hosting)

### Troubleshooting

1. Check Firebase Console logs
2. Verify environment variables
3. Test functions locally first
4. Check Firestore rules
5. Review CORS configuration

## 🎉 Success!

Once deployed, your Smart Content Agent will be available at:
`https://your-project.web.app`

Features available:
- ✅ Serverless backend with Firebase Functions
- ✅ Static frontend with Firebase Hosting
- ✅ Scalable and cost-effective
- ✅ Global CDN and HTTPS
- ✅ Real-time monitoring and logging

**Your AI-powered content analysis platform is now live on Firebase!** 🚀
