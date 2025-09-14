#!/bin/bash

# Mentra Reality Pipeline Web Application Startup Script

echo "🚀 Starting Mentra Reality Pipeline Web Application..."
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "web_app.py" ]; then
    echo "❌ web_app.py not found. Please run this script from the project root directory."
    exit 1
fi

# Install requirements if needed
echo "📦 Installing/updating requirements..."
python3 -m pip install -r requirements_web.txt

# Check if installation was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements. Please check your Python environment."
    exit 1
fi

echo "✅ Requirements installed successfully"
echo ""

# Start the web application
echo "🌐 Starting web server..."
echo "📱 Web interface will be available at: http://localhost:5000"
echo "🛑 Press Ctrl+C to stop the server"
echo "=================================================="

# Run the web app
python3 web_app.py
