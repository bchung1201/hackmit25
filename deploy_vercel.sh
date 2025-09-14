#!/bin/bash

# Vercel Deployment Script for Mentra Reality Pipeline

echo "🚀 Deploying Mentra Reality Pipeline to Vercel..."
echo "=================================================="

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Please initialize git first:"
    echo "   git init"
    echo "   git add ."
    echo "   git commit -m 'Initial commit'"
    exit 1
fi

# Check if we have the required files
required_files=("index.html" "styles.css" "script.js" "vercel.json" "package.json")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing required file: $file"
        exit 1
    fi
done

echo "✅ All required files found"

# Check git status
if [ -n "$(git status --porcelain)" ]; then
    echo "📝 Staging changes..."
    git add .
    
    echo "💾 Committing changes..."
    git commit -m "Add Vercel deployment configuration"
    
    echo "🌐 Pushing to GitHub..."
    git push origin main
else
    echo "✅ No changes to commit"
fi

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."
vercel --prod

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Successfully deployed to Vercel!"
    echo "📱 Your site is now live!"
    echo ""
    echo "🔧 To manage your deployment:"
    echo "   - Visit: https://vercel.com/dashboard"
    echo "   - Or run: vercel"
    echo ""
    echo "🔄 Future deployments:"
    echo "   - Just push to GitHub: git push origin main"
    echo "   - Or deploy manually: vercel --prod"
else
    echo "❌ Deployment failed. Please check the error messages above."
    exit 1
fi
