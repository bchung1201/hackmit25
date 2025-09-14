#!/bin/bash

# Mentra Real-Time Stream Starter Script

echo "🚀 Starting Mentra Real-Time Stream Server..."
echo "==========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Creating .env from .env.example..."
    
    # Create .env from example if it exists
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env file. Please edit it with your credentials:"
        echo "   - PACKAGE_NAME: Get from https://console.mentra.glass/"
        echo "   - MENTRAOS_API_KEY: Get from https://console.mentra.glass/"
        echo ""
        echo "After updating .env, run this script again."
        exit 1
    else
        echo "❌ No .env.example file found. Please create a .env file with:"
        echo "PACKAGE_NAME=your-package-name"
        echo "MENTRAOS_API_KEY=your-api-key"
        echo "PORT=3000"
        exit 1
    fi
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    bun install
fi

# Start the server
echo ""
echo "Starting server..."
echo "Open http://localhost:3000 in your browser"
echo "Press Ctrl+C to stop"
echo ""

bun run dev
