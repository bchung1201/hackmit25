#!/usr/bin/env python3
"""
Mentra Reality Pipeline Web Application
Integrates frontend (HTML/CSS/JS) with backend (3D reconstruction + emotion detection)
"""

import os
import asyncio
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import tempfile
import json
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
import time
import threading
import queue

# Import your existing pipelines
from room_reconstruction_3d import Room3DReconstructionPipeline
from video_emotion_summary import VideoEmotionSummary
from emotion_room_mapper import EmotionRoomMapper
from emotion_detection.emotion_processor import EmotionProcessor
from emotion_detection.face_detector import FaceDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Global processing state
processing_queue = queue.Queue()
processing_status = {}

# Initialize pipelines
room_pipeline = None
emotion_pipeline = None
emotion_room_mapper = None

def initialize_pipelines():
    """Initialize the processing pipelines"""
    global room_pipeline, emotion_pipeline, emotion_room_mapper
    
    try:
        # Initialize 3D reconstruction pipeline
        room_config = {
            'frame_interval': 30,
            'use_modal': False,
            'output_dir': 'outputs/3d_reconstruction',
            'rooms': ['Room_1']
        }
        room_pipeline = Room3DReconstructionPipeline(room_config)
        
        # Initialize emotion detection
        emotion_processor = EmotionProcessor()
        face_detector = FaceDetector()
        emotion_room_mapper = EmotionRoomMapper()
        
        logger.info("✅ All pipelines initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline initialization failed: {e}")
        return False

# Initialize pipelines on startup
pipeline_ready = initialize_pipelines()

@app.route('/')
def index():
    """Serve the main frontend page"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """API endpoint to check system status"""
    return jsonify({
        'status': 'online',
        'pipeline_ready': pipeline_ready,
        'timestamp': time.time()
    })

@app.route('/api/demo_mode')
def api_demo_mode():
    """API endpoint for demo mode detection"""
    return jsonify({
        'pipeline_available': pipeline_ready,
        'demo_mode': not pipeline_ready,
        'features': {
            '3d_reconstruction': pipeline_ready,
            'emotion_detection': pipeline_ready,
            'room_highlighting': pipeline_ready
        }
    })

@app.route('/api/process_video', methods=['POST'])
def api_process_video():
    """API endpoint to process uploaded video"""
    if not pipeline_ready:
        return jsonify({
            'success': False,
            'error': 'Pipeline not ready'
        }), 500
    
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No video file selected'
            }), 400
        
        # Save uploaded video
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)
        
        # Generate unique job ID
        job_id = f"job_{int(time.time())}"
        processing_status[job_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting video processing...',
            'results': None,
            'error': None
        }
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_video_background,
            args=(job_id, video_path, filename)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Video processing started'
        })
        
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/processing_status/<job_id>')
def api_processing_status(job_id):
    """API endpoint to check processing status"""
    if job_id not in processing_status:
        return jsonify({
            'success': False,
            'error': 'Job not found'
        }), 404
    
    return jsonify({
        'success': True,
        'status': processing_status[job_id]
    })

@app.route('/api/results/<job_id>')
def api_results(job_id):
    """API endpoint to get processing results"""
    if job_id not in processing_status:
        return jsonify({
            'success': False,
            'error': 'Job not found'
        }), 404
    
    job_data = processing_status[job_id]
    
    if job_data['status'] != 'completed':
        return jsonify({
            'success': False,
            'error': 'Job not completed yet'
        }), 400
    
    return jsonify({
        'success': True,
        'results': job_data['results']
    })

@app.route('/api/download/<job_id>/<file_type>')
def api_download(job_id, file_type):
    """API endpoint to download result files"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    job_data = processing_status[job_id]
    if job_data['status'] != 'completed':
        return jsonify({'error': 'Job not completed'}), 400
    
    results = job_data['results']
    
    # Map file types to actual file paths
    file_mapping = {
        '3d_scene': results.get('3d_scene_path'),
        'emotion_map': results.get('emotion_map_path'),
        'video_processed': results.get('processed_video_path'),
        'screenshot': results.get('screenshot_path')
    }
    
    if file_type not in file_mapping or not file_mapping[file_type]:
        return jsonify({'error': 'File not found'}), 404
    
    file_path = file_mapping[file_type]
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found on disk'}), 404
    
    return send_file(file_path, as_attachment=True)

def process_video_background(job_id: str, video_path: str, filename: str):
    """Background processing function"""
    try:
        # Update status
        processing_status[job_id]['message'] = 'Initializing 3D reconstruction...'
        processing_status[job_id]['progress'] = 10
        
        # Run 3D reconstruction
        room_result = asyncio.run(room_pipeline.process_video(
            video_path=video_path,
            room_names=['Room_1'],
            use_modal=False
        ))
        
        processing_status[job_id]['message'] = 'Running emotion detection...'
        processing_status[job_id]['progress'] = 50
        
        # Run emotion detection
        emotion_summary = VideoEmotionSummary(video_path, 'outputs/emotion_summary')
        emotion_result = asyncio.run(emotion_summary.process_video(
            frame_interval=30,
            simulate_movement=True
        ))
        
        processing_status[job_id]['message'] = 'Generating outputs...'
        processing_status[job_id]['progress'] = 80
        
        # Prepare results
        results = {
            'video_info': room_result.get('video_info', {}),
            '3d_reconstruction': {
                'rooms_reconstructed': room_result.get('scene_statistics', {}).get('room_count', 0),
                'furniture_reconstructed': room_result.get('scene_statistics', {}).get('furniture_count', 0),
                'processing_time': room_result.get('processing_time', 0)
            },
            'emotion_detection': {
                'total_emotions': len(emotion_result.get('emotion_data', [])),
                'rooms_analyzed': len(emotion_result.get('room_emotions', {})),
                'processing_time': emotion_result.get('processing_time', 0)
            },
            'files': {
                '3d_scene_path': 'outputs/3d_reconstruction/complete_scene_complete.ply',
                'emotion_map_path': 'outputs/emotion_summary/emotion_summary_map.png',
                'screenshot_path': 'outputs/3d_reconstruction/scene_screenshot.png'
            }
        }
        
        # Update final status
        processing_status[job_id].update({
            'status': 'completed',
            'progress': 100,
            'message': 'Processing completed successfully!',
            'results': results
        })
        
        logger.info(f"✅ Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}")
        processing_status[job_id].update({
            'status': 'failed',
            'progress': 0,
            'message': f'Processing failed: {str(e)}',
            'error': str(e)
        })

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'pipeline_ready': pipeline_ready
    })

# Serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create templates directory and copy HTML
    os.makedirs('templates', exist_ok=True)
    
    # Copy HTML file to templates
    with open('index.html', 'r') as f:
        html_content = f.read()
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)
    
    # Create static directory and copy CSS/JS
    os.makedirs('static', exist_ok=True)
    
    # Copy CSS file
    with open('styles.css', 'r') as f:
        css_content = f.read()
    
    with open('static/styles.css', 'w') as f:
        f.write(css_content)
    
    # Copy JS file
    with open('script.js', 'r') as f:
        js_content = f.read()
    
    with open('static/script.js', 'w') as f:
        f.write(js_content)
    
    # Update HTML to use Flask static files
    html_content = html_content.replace('href="styles.css"', 'href="{{ url_for(\'static\', filename=\'styles.css\') }}"')
    html_content = html_content.replace('src="script.js"', 'src="{{ url_for(\'static\', filename=\'script.js\') }}"')
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)
    
    logger.info("🚀 Starting Mentra Reality Pipeline Web App...")
    logger.info(f"📊 Pipeline ready: {pipeline_ready}")
    logger.info("🌐 Web interface available at: http://localhost:5000")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
