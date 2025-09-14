# 3D Room Reconstruction System

A comprehensive system for reconstructing 3D rooms and furniture from video using computer vision and machine learning techniques.

## 🎯 Features

### ✅ **3D Room Reconstruction**
- Multi-view stereo reconstruction
- Point cloud generation and cleaning
- Mesh generation with Poisson reconstruction
- Room segmentation and alignment

### ✅ **Furniture Detection & Reconstruction**
- YOLO-based furniture detection
- Object tracking across frames
- 3D object reconstruction
- Furniture classification and placement

### ✅ **Scene Assembly**
- Complete 3D scene assembly
- Room and furniture alignment
- Spatial relationship modeling
- Scene optimization and cleaning

### ✅ **3D Visualization**
- Interactive Open3D viewer
- HTML/WebGL 3D viewer
- Screenshot capture
- Export to standard 3D formats

### ✅ **Modal Cloud Integration**
- GPU-accelerated processing
- Distributed cloud computing
- Scalable reconstruction pipeline
- Cost-effective processing

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Demo
```bash
python demo_3d_reconstruction.py
```

### 3. Process Your Video
```bash
python room_reconstruction_3d.py your_video.mp4
```

## 📁 File Structure

```
reconstruction_3d/
├── __init__.py                    # Package initialization
├── room_reconstructor.py          # 3D room reconstruction
├── furniture_detector.py          # YOLO furniture detection
├── object_reconstructor.py        # 3D object reconstruction
├── scene_assembler.py             # Scene assembly
├── visualizer.py                  # 3D visualization
└── modal_integration.py           # Modal cloud processing

room_reconstruction_3d.py          # Main pipeline script
demo_3d_reconstruction.py          # Interactive demo
test_3d_reconstruction.py          # Test script
README_3D_RECONSTRUCTION.md        # This documentation
```

## 🎬 Usage Examples

### Basic Usage
```python
from room_reconstruction_3d import Room3DReconstructionPipeline

# Create pipeline
pipeline = Room3DReconstructionPipeline()

# Process video
result = await pipeline.process_video(
    video_path="your_video.mp4",
    room_names=["Living Room", "Kitchen", "Bedroom"]
)
```

### Advanced Usage
```python
# Custom configuration
config = {
    'frame_interval': 15,  # Process every 15th frame
    'use_modal': True,     # Use Modal cloud processing
    'output_dir': 'my_3d_outputs',
    'room_reconstruction': {
        'voxel_size': 0.005,
        'enable_mesh_generation': True
    },
    'furniture_detection': {
        'confidence_threshold': 0.6,
        'enable_tracking': True
    }
}

pipeline = Room3DReconstructionPipeline(config)
result = await pipeline.process_video("video.mp4")
```

### Modal Cloud Processing
```python
# Use Modal for GPU acceleration
result = await pipeline.process_video(
    video_path="video.mp4",
    use_modal=True
)
```

## 📊 Output Files

### 3D Models
- **Complete scene** (PLY format)
- **Individual rooms** (PLY format)
- **Furniture objects** (PLY format)
- **Meshes** (PLY format)

### Visualization
- **3D viewer** (HTML/WebGL)
- **Screenshots** (PNG format)
- **Interactive viewer** (Open3D)

### Data
- **Scene statistics** (JSON)
- **Room metadata** (JSON)
- **Furniture inventory** (JSON)

## 🏗️ Technical Details

### 3D Reconstruction Pipeline
```
Video Frames → SLAM Poses → Multi-view Stereo → 3D Point Cloud
     ↓
Furniture Detection → Object Tracking → 3D Object Reconstruction
     ↓
Scene Assembly → 3D Visualization → Export
```

### Key Components

#### 1. **Room Reconstruction**
- **Multi-view stereo** for 3D point generation
- **Feature matching** with ORB detector
- **Triangulation** using camera poses
- **Point cloud cleaning** and optimization

#### 2. **Furniture Detection**
- **YOLO v8** for object detection
- **Object tracking** across frames
- **Furniture classification** (chairs, tables, beds, etc.)
- **Bounding box refinement**

#### 3. **3D Object Reconstruction**
- **Multi-view stereo** for individual objects
- **Object tracking** and association
- **3D model generation** with meshes
- **Object placement** in scene

#### 4. **Scene Assembly**
- **Room alignment** in common coordinate system
- **Furniture placement** in appropriate rooms
- **Spatial optimization** and cleaning
- **Complete scene generation**

## ⚙️ Configuration

### Room Reconstruction
```python
room_config = {
    'voxel_size': 0.01,              # Point cloud downsampling
    'enable_mesh_generation': True,   # Generate 3D meshes
    'stereo_num_disparities': 64,    # Stereo matching
    'stereo_block_size': 15,         # Stereo block size
    'max_features': 1000,            # Feature detection
    'min_matches': 50                # Minimum feature matches
}
```

### Furniture Detection
```python
furniture_config = {
    'confidence_threshold': 0.5,     # Detection confidence
    'nms_threshold': 0.4,            # Non-maximum suppression
    'max_detections': 100,           # Maximum detections per frame
    'enable_tracking': True,         # Object tracking
    'input_size': 640                # YOLO input size
}
```

### Scene Assembly
```python
scene_config = {
    'enable_furniture_placement': True,  # Place furniture in rooms
    'enable_room_alignment': True,       # Align rooms
    'furniture_placement_threshold': 0.1, # Placement distance
    'enable_scene_optimization': True    # Optimize scene
}
```

## 🚀 Modal Integration

### Cloud Processing
```python
# Modal functions for GPU acceleration
@modal.function(gpu="A10G")
def process_video_frames_modal(video_data, frame_interval):
    # Process video on GPU
    return processed_frames

@modal.function(gpu="T4")
def detect_furniture_modal(frames):
    # YOLO detection on GPU
    return furniture_detections

@modal.function(gpu="A10G")
def reconstruct_room_3d_modal(frames, poses, room_name):
    # 3D reconstruction on GPU
    return room_model
```

### Performance Benefits
- **GPU acceleration** for 3D reconstruction
- **Distributed processing** for large videos
- **Scalable resources** based on needs
- **Cost-effective** cloud computing

## 📈 Performance

### Processing Times
- **Local (CPU)**: 10-30 minutes for 10-minute video
- **Modal (GPU)**: 5-15 minutes for 10-minute video
- **Frame interval**: 30 (fast) to 10 (detailed)

### Memory Usage
- **RAM**: 4-8 GB typical
- **GPU**: 8-16 GB VRAM (Modal)
- **Storage**: 100-500 MB per video

### Quality Settings
- **Fast**: Frame interval 60, low resolution
- **Medium**: Frame interval 30, medium resolution
- **High**: Frame interval 10, high resolution

## 🔧 Troubleshooting

### Common Issues

1. **"No furniture detected"**
   - Lower confidence threshold
   - Check video quality
   - Ensure good lighting

2. **"Poor 3D reconstruction"**
   - Increase frame interval
   - Check camera movement
   - Verify feature detection

3. **"Modal processing failed"**
   - Check internet connection
   - Verify Modal setup
   - Check video file size

4. **"Memory errors"**
   - Reduce frame interval
   - Use smaller video
   - Enable Modal processing

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Use Cases

### 1. **Real Estate**
- 3D property tours
- Virtual staging
- Space planning

### 2. **Interior Design**
- Room visualization
- Furniture placement
- Design planning

### 3. **Architecture**
- Space analysis
- Room layout optimization
- 3D documentation

### 4. **Research**
- 3D scene understanding
- Object recognition
- Spatial analysis

## 🚀 Future Enhancements

### Planned Features
- **Real-time SLAM** integration
- **More furniture types** detection
- **3D object completion** with AI
- **VR/AR integration**

### Advanced Features
- **Semantic segmentation** of rooms
- **Material recognition** and texturing
- **Lighting estimation** and rendering
- **Interactive 3D editing**

## 📝 License

This project is part of the Mentra pipeline and follows the same licensing terms.

## 🤝 Contributing

Contributions are welcome! Please see the main project README for contribution guidelines.

---

**Note**: This system is designed for demonstration and research purposes. For production use, ensure proper privacy and security measures are in place.
