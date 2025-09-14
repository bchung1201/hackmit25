# Video Emotion Summary System

A comprehensive system that processes videos to detect emotions and generates a **single summary map** showing overall emotions per room. This system uses multiple emotion detection models and room-aware processing to provide accurate, detailed emotion analysis.

## 🎯 Key Features

### ✅ **Single Summary Map**
- **ONE map** showing overall emotions per room (not per frame)
- Room colors represent dominant emotions
- Intensity based on confidence scores
- Emotion trends and statistics

### ✅ **Multi-Model Emotion Detection**
- **EmoNet** (8 emotions) - Primary model
- **FER2013** (7 emotions) - Backup model  
- **Simple CNN** (5 emotions) - Fallback model
- **MediaPipe** (Face mesh) - Alternative model
- **Ensemble prediction** - Combines multiple models

### ✅ **Room-Aware Processing**
- Tracks which room the person is in for each frame
- Aggregates emotions per room over time
- Calculates room-level statistics
- Simulates room movement for demo purposes

### ✅ **Comprehensive Statistics**
- Overall mood across all rooms
- Most/least emotional rooms
- Emotion distribution per room
- Confidence scores and trends
- Processing performance metrics

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Demo
```bash
python demo_video_emotion_summary.py
```

### 3. Process Your Video
```bash
python video_emotion_summary.py your_video.mp4
```

## 📁 File Structure

```
emotion_detection/
├── multi_model_detector.py      # Multi-model emotion detection
├── room_aware_processor.py      # Room-aware emotion processing
├── face_detector.py             # Face detection
├── emonet_detector.py           # EmoNet model
└── emotion_processor.py         # Emotion processing orchestration

room_highlighting/
├── emotion_summary_generator.py # Summary map generation
├── roborock_parser.py           # Map data parsing
└── room_highlighter.py          # Room highlighting

video_emotion_summary.py         # Main video processing script
demo_video_emotion_summary.py    # Interactive demo
test_video_emotion_summary.py    # Test script
```

## 🎬 Usage Examples

### Basic Usage
```python
from video_emotion_summary import VideoEmotionSummary

# Create processor
processor = VideoEmotionSummary("your_video.mp4", "output_dir")

# Process video
results = await processor.process_video(
    frame_interval=30,  # Process every 30th frame
    simulate_movement=True
)

# Get summary
emotion_summary = results['emotion_summary']
print(f"Overall mood: {emotion_summary.overall_mood}")
print(f"Most emotional room: {emotion_summary.most_emotional_room}")
```

### Advanced Usage
```python
from emotion_detection.room_aware_processor import RoomAwareEmotionProcessor
from room_highlighting.emotion_summary_generator import EmotionSummaryMapGenerator

# Create processor with custom room positions
room_positions = {
    "Living Room": (300, 200),
    "Kitchen": (500, 300),
    "Bedroom": (200, 400)
}

processor = RoomAwareEmotionProcessor(room_positions)

# Process frame
result = await processor.process_frame_with_room(
    frame=frame,
    position=(350, 250),
    frame_id=0,
    timestamp=0.0
)

# Generate summary map
map_generator = EmotionSummaryMapGenerator()
emotion_summary = processor.get_room_emotion_summary()
summary_map = map_generator.generate_summary_map(emotion_summary)
```

## 📊 Output Files

### 1. **emotion_summary_map.png**
- Main summary map showing room emotions
- Color-coded rooms by dominant emotion
- Statistics overlay
- Confidence scores and trends

### 2. **emotion_data.json**
- Raw emotion data in JSON format
- Room-by-room emotion statistics
- Processing metadata

### 3. **emotion_report.txt**
- Detailed text report
- Room-by-room analysis
- Processing statistics
- Model information

## 🏠 Room Emotion Mapping

### Default Room Layout
```
┌─────────────┬─────────────┬─────────────┐
│  Bar Room   │ Laundry Rm  │ Game Room   │
│   (175,225) │  (400,125)  │  (650,125)  │
├─────────────┼─────────────┼─────────────┤
│             │  Hallway    │             │
│             │  (400,225)  │             │
├─────────────┼─────────────┼─────────────┤
│             │             │ Living Room │
│             │             │  (650,375)  │
└─────────────┴─────────────┴─────────────┘
```

### Emotion Color Mapping
- **Happy**: Green
- **Sad**: Blue  
- **Angry**: Red
- **Fear**: Purple
- **Surprise**: Orange
- **Disgust**: Brown
- **Neutral**: Gray
- **Excited**: Yellow

## 🤖 Model Details

### EmoNet (Primary)
- **Emotions**: 8 classes
- **Output**: Emotion + Valence + Arousal
- **Confidence**: High accuracy
- **Use Case**: Primary emotion detection

### FER2013 (Backup)
- **Emotions**: 7 classes
- **Output**: Emotion classification
- **Confidence**: Medium accuracy
- **Use Case**: When EmoNet fails

### Simple CNN (Fallback)
- **Emotions**: 5 classes
- **Output**: Basic emotion classification
- **Confidence**: Lower accuracy
- **Use Case**: When other models fail

### MediaPipe (Alternative)
- **Emotions**: Landmark-based
- **Output**: Basic emotion detection
- **Confidence**: Medium accuracy
- **Use Case**: Alternative approach

## ⚙️ Configuration

### Frame Processing
```python
# Process every 30th frame (faster)
frame_interval = 30

# Process every 10th frame (more detailed)
frame_interval = 10

# Process every frame (slowest, most detailed)
frame_interval = 1
```

### Map Configuration
```python
from room_highlighting.emotion_summary_generator import EmotionMapConfig

config = EmotionMapConfig(
    map_width=1200,
    map_height=800,
    show_statistics=True,
    show_emotion_distribution=True,
    show_confidence_scores=True,
    show_trends=True
)
```

## 📈 Performance

### Typical Processing Times
- **Fast Mode** (60fps interval): ~2-5 minutes for 10-minute video
- **Medium Mode** (30fps interval): ~5-10 minutes for 10-minute video  
- **Slow Mode** (10fps interval): ~15-30 minutes for 10-minute video

### Memory Usage
- **RAM**: 2-4 GB typical
- **GPU**: Optional (CUDA acceleration)
- **Storage**: 50-200 MB per video

## 🔧 Troubleshooting

### Common Issues

1. **"No emotion detection models available"**
   - Install missing dependencies: `pip install -r requirements.txt`
   - Check PyTorch installation

2. **"Video file not found"**
   - Verify video path exists
   - Check file permissions

3. **"Low emotion detection accuracy"**
   - Try different frame intervals
   - Check video quality
   - Ensure good lighting

4. **"Room detection not working"**
   - Verify room positions are set correctly
   - Check position coordinates

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Use Cases

### 1. **Home Security**
- Monitor emotional states in different rooms
- Detect unusual behavior patterns
- Generate room-level emotion reports

### 2. **Healthcare**
- Track patient emotional well-being
- Monitor therapy progress
- Generate emotion trend reports

### 3. **Research**
- Study emotion patterns in different environments
- Analyze room-based emotional responses
- Generate statistical emotion data

### 4. **Smart Home**
- Adjust room lighting based on emotions
- Play appropriate music per room
- Optimize room comfort settings

## 🚀 Future Enhancements

### Planned Features
- **Real-time SLAM integration** for accurate room tracking
- **More emotion models** (FER+, AffectNet, etc.)
- **3D emotion mapping** with depth information
- **Emotion prediction** based on room transitions
- **Mobile app** for real-time monitoring

### Integration Options
- **Home Assistant** integration
- **IoT device** control
- **Cloud processing** with Modal Labs
- **Real-time streaming** support

## 📝 License

This project is part of the Mentra pipeline and follows the same licensing terms.

## 🤝 Contributing

Contributions are welcome! Please see the main project README for contribution guidelines.

---

**Note**: This system is designed for demonstration and research purposes. For production use, ensure proper privacy and security measures are in place.
