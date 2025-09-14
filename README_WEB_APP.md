# Mentra Reality Pipeline Web Application

A comprehensive web interface that integrates the frontend (HTML/CSS/JS) with the backend Python pipelines for 3D reconstruction and emotion detection.

## 🎯 Features

### ✅ **Web Interface**
- Beautiful, responsive frontend design
- Real-time video upload and processing
- Interactive demo with live progress updates
- Download results (3D scenes, emotion maps, screenshots)

### ✅ **Backend Integration**
- Flask web server with REST API
- Integration with 3D reconstruction pipeline
- Integration with emotion detection pipeline
- Background processing with job queue
- Real-time status updates

### ✅ **API Endpoints**
- `/api/status` - System status check
- `/api/demo_mode` - Demo mode detection
- `/api/process_video` - Video processing endpoint
- `/api/processing_status/<job_id>` - Job status check
- `/api/results/<job_id>` - Get processing results
- `/api/download/<job_id>/<file_type>` - Download result files

## 🚀 Quick Start

### Option 1: Using the startup script (Recommended)
```bash
./run_web_app.sh
```

### Option 2: Manual startup
```bash
# Install requirements
pip install -r requirements_web.txt

# Start the web app
python web_app.py
```

### Option 3: Using the Python startup script
```bash
python start_web_app.py
```

## 🌐 Usage

1. **Open your browser** and go to `http://localhost:5000`
2. **Upload a video** by clicking the upload area or dragging & dropping
3. **Click "Process Video"** to start the analysis
4. **Watch the progress** as the system processes your video
5. **View results** including 3D reconstruction, emotion analysis, and floor maps
6. **Download files** using the download buttons

## 📁 File Structure

```
web_app.py                 # Main Flask application
templates/
├── index.html            # Main HTML template
static/
├── styles.css            # CSS styles
└── script.js             # JavaScript functionality
requirements_web.txt      # Web app dependencies
start_web_app.py          # Python startup script
run_web_app.sh           # Bash startup script
README_WEB_APP.md        # This documentation
```

## 🔧 Configuration

The web app automatically configures itself with:
- **Upload folder**: `uploads/` (for uploaded videos)
- **Output folder**: `outputs/` (for processed results)
- **Max file size**: 500MB
- **Port**: 5000
- **Host**: 0.0.0.0 (accessible from any IP)

## 📊 Processing Pipeline

1. **Video Upload** → User uploads video file
2. **3D Reconstruction** → Room and furniture detection
3. **Emotion Detection** → Facial expression analysis
4. **Floor Map Generation** → Room-by-room emotion mapping
5. **Results Display** → Interactive visualization
6. **File Downloads** → Export results

## 🛠️ API Usage

### Check System Status
```bash
curl http://localhost:5000/api/status
```

### Process Video
```bash
curl -X POST -F "video=@your_video.mp4" http://localhost:5000/api/process_video
```

### Check Processing Status
```bash
curl http://localhost:5000/api/processing_status/job_1234567890
```

### Download Results
```bash
curl http://localhost:5000/api/download/job_1234567890/3d_scene
curl http://localhost:5000/api/download/job_1234567890/emotion_map
curl http://localhost:5000/api/download/job_1234567890/screenshot
```

## 🎨 Frontend Features

- **Responsive Design** - Works on desktop and mobile
- **Drag & Drop Upload** - Easy video file upload
- **Real-time Progress** - Live processing updates
- **Interactive Results** - Visual data representation
- **Download Management** - Easy file downloads
- **Error Handling** - User-friendly error messages

## 🔍 Troubleshooting

### Common Issues

1. **Port 5000 already in use**
   ```bash
   # Kill process using port 5000
   lsof -ti:5000 | xargs kill -9
   ```

2. **Missing dependencies**
   ```bash
   pip install -r requirements_web.txt
   ```

3. **Video processing fails**
   - Check video format (MP4, MOV, AVI supported)
   - Ensure video file size < 500MB
   - Check server logs for detailed error messages

4. **Frontend not loading**
   - Clear browser cache
   - Check if static files are being served correctly
   - Verify Flask is running on correct port

### Debug Mode

To run in debug mode:
```bash
export FLASK_DEBUG=1
python web_app.py
```

## 📈 Performance

- **Concurrent Processing** - Multiple videos can be processed simultaneously
- **Background Jobs** - Non-blocking video processing
- **Memory Efficient** - Streaming video processing
- **Caching** - Static file caching for better performance

## 🔒 Security

- **File Validation** - Only video files accepted
- **Size Limits** - 500MB maximum file size
- **CORS Enabled** - Cross-origin requests supported
- **Error Handling** - Secure error messages

## 🚀 Deployment

For production deployment:

1. **Use a production WSGI server** (e.g., Gunicorn)
2. **Set up reverse proxy** (e.g., Nginx)
3. **Configure SSL** for HTTPS
4. **Set up monitoring** and logging
5. **Use environment variables** for configuration

Example production setup:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_app:app
```

## 📝 License

This project is part of the HackMIT 2025 Mentra Reality Pipeline.
