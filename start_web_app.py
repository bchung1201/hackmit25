#!/usr/bin/env python3
"""
Startup script for Mentra Reality Pipeline Web Application
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install web app requirements"""
    print("📦 Installing web application requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_web.txt"
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_dependencies():
    """Check if all required files exist"""
    required_files = [
        "web_app.py",
        "room_reconstruction_3d.py", 
        "video_emotion_summary.py",
        "emotion_room_mapper.py",
        "index.html",
        "styles.css", 
        "script.js"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found")
    return True

def main():
    """Main startup function"""
    print("🚀 Starting Mentra Reality Pipeline Web Application...")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please ensure all files are present.")
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please check your Python environment.")
        return
    
    print("\n🌐 Starting web server...")
    print("📱 Web interface will be available at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Start the web app
    try:
        from web_app import app
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
