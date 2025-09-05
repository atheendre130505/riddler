#!/bin/bash

# Firebase Deployment Script for Smart Content Agent
echo "🔥 Smart Content Agent - Firebase Deployment"
echo "============================================="

# Check if Firebase CLI is installed
if ! command -v firebase &> /dev/null; then
    echo "❌ Firebase CLI not found. Installing..."
    npm install -g firebase-tools
fi

# Check if user is logged in
if ! firebase projects:list &> /dev/null; then
    echo "🔐 Please login to Firebase:"
    firebase login
fi

echo ""
echo "📋 Deployment Checklist:"
echo "1. ✅ Firebase CLI installed"
echo "2. ✅ User authenticated"
echo "3. 🔄 Initializing Firebase project..."

# Initialize Firebase if not already done
if [ ! -f ".firebaserc" ]; then
    echo "Initializing Firebase project..."
    firebase init --project smart-content-agent
fi

echo ""
echo "🚀 Deploying to Firebase..."

# Deploy Functions and Hosting
firebase deploy

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Your app is now available at:"
echo "   https://smart-content-agent.web.app"
echo ""
echo "📊 Monitor your deployment:"
echo "   https://console.firebase.google.com"
echo ""
echo "🔧 Available endpoints:"
echo "   GET  /api/health"
echo "   POST /api/upload"
echo "   POST /api/process-url"
echo ""
echo "📚 For detailed instructions, see FIREBASE_DEPLOYMENT.md"
