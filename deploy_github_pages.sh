#!/bin/bash

# GitHub Pages Deployment Script for Mentra Reality Pipeline

echo "🚀 Deploying Mentra Reality Pipeline to GitHub Pages..."
echo "=================================================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Please initialize git first:"
    echo "   git init"
    echo "   git add ."
    echo "   git commit -m 'Initial commit'"
    exit 1
fi

# Check if we have the required files
required_files=("index.html" "styles.css" "script.js")
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
    git commit -m "Update GitHub Pages demo"
else
    echo "✅ No changes to commit"
fi

# Push to GitHub
echo "🌐 Pushing to GitHub..."
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Successfully deployed to GitHub Pages!"
    echo "📱 Your site should be available at:"
    echo "   https://$(git config --get remote.origin.url | sed 's/.*github.com[:/]\([^/]*\)\/\([^.]*\).*/\1.github.io\/\2/')"
    echo ""
    echo "⏰ Note: It may take a few minutes for changes to appear on GitHub Pages"
    echo "🔧 To check deployment status, go to:"
    echo "   https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\([^/]*\)\/\([^.]*\).*/\1\/\2/')/actions"
else
    echo "❌ Failed to push to GitHub. Please check your git configuration."
    exit 1
fi
