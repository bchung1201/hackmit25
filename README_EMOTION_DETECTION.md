# Emotion Detection and Room Highlighting System

This document describes the complete emotion detection and room highlighting system integrated into the Mentra pipeline. The system detects human facial expressions in real-time and dynamically highlights rooms on a Roborock map based on the detected emotions.

## 🎯 Overview

The system solves the original problem where **objects were being detected as expressions** by implementing proper **human face detection** and **facial expression recognition** using the [EmoNet](https://github.com/face-analysis/emonet) model.

### Key Features

- **Real-time Face Detection**: Uses OpenCV and MTCNN for robust face detection
- **Facial Expression Recognition**: EmoNet-based emotion detection with 8 emotion classes
- **Room Highlighting**: Dynamic room highlighting on Roborock maps based on emotions
- **Emotion Mapping**: Intelligent mapping of emotions to room highlighting states
- **Real-time Integration**: Seamlessly integrated with the existing Mentra pipeline
- **Visual Feedback**: Generates highlighted map images for visualization

## 🏗️ Architecture

```
Mentra Glasses → Video Stream → Face Detection → EmoNet → Emotion Analysis
                                                                    ↓
Roborock Map ← Room Highlighter ← Emotion Mapper ← Expression Results
```

### Components

1. **Face Detection** (`emotion_detection/face_detector.py`)
   - OpenCV Haar Cascades for basic face detection
   - MTCNN support for advanced face detection
   - Face region extraction and preprocessing

2. **Emotion Detection** (`emotion_detection/emonet_detector.py`)
   - EmoNet model integration for facial expression recognition
   - 8 emotion classes: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt
   - Valence and arousal detection for nuanced emotional states
   - Confidence scoring and emotion intensity calculation

3. **Emotion Processing** (`emotion_detection/emotion_processor.py`)
   - Main emotion processing pipeline
   - Frame-by-frame emotion detection
   - Emotion trend analysis and statistics
   - Multi-face support

4. **Room Highlighting** (`room_highlighting/`)
   - Roborock map parser for JSON/XML map formats
   - Room highlighting engine with animation effects
   - Dynamic color mapping based on emotions
   - Map image generation

5. **Emotion-Room Mapping** (`emotion_room_mapper.py`)
   - Maps detected emotions to room highlighting states
   - Configurable highlighting rules
   - Context-aware room selection
   - Real-time map updates

## 🎭 Emotion Classes and Room Mapping

### Emotion Classes (8-class EmoNet)

| Emotion | Description | Room Highlighting |
|---------|-------------|-------------------|
| **Happy** | Joy, contentment | Game Room, Bar Room, Living Room (bright green, glow effect) |
| **Excited** | High arousal, enthusiasm | Game Room, Bar Room (bright yellow, pulse effect) |
| **Sad** | Sorrow, melancholy | Living Room, Bedroom (deep blue, fade effect) |
| **Angry** | Anger, frustration | Living Room, Bedroom (red, static highlighting) |
| **Fear** | Fear, anxiety | Living Room, Bedroom, Bathroom (purple, pulse effect) |
| **Surprise** | Surprise, astonishment | Game Room, Living Room (orange, glow effect) |
| **Disgust** | Disgust, aversion | Bathroom, Kitchen, Laundry Room (brown, fade effect) |
| **Contempt** | Contempt, disdain | Living Room, Office (dim gray, static highlighting) |

### Additional Categories

- **Positive Mood**: Brightens all social areas (Game Room, Bar Room, Living Room, Dining Room)
- **Negative Mood**: Highlights private spaces (Bedroom, Living Room, Bathroom)
- **High Arousal**: Emphasizes active areas (Game Room, Bar Room, Living Room)
- **Mixed Emotions**: Subtle highlighting of common areas (Living Room, Hallway)
- **Neutral**: Minimal highlighting for calm states

## 🚀 Usage

### Basic Usage

```python
from emotion_room_mapper import EmotionRoomMapper

# Initialize the system
mapper = EmotionRoomMapper()

# Process a video frame
result = await mapper.process_frame(
    frame=video_frame,
    frame_id=0,
    timestamp=0.0
)

# Get highlighted map image
map_image = mapper.get_highlighted_map_image(1000, 600)
```

### Integration with Mentra Pipeline

The system is automatically integrated into the `EnhancedMentraVideoPipeline`:

```python
from enhanced_video_pipeline import EnhancedMentraVideoPipeline
from config import DEFAULT_CONFIG

# Create pipeline with emotion detection enabled
config = DEFAULT_CONFIG.to_dict()
config['enable_emotion_detection'] = True
config['enable_room_highlighting'] = True

pipeline = EnhancedMentraVideoPipeline(config, use_mock_components=True)

# Start pipeline
await pipeline.start()

# Get emotion status
emotion_status = await pipeline.get_emotion_status()
print(f"Current emotion: {emotion_status['current_emotion']}")
print(f"Highlighted rooms: {emotion_status['highlighted_rooms']}")

# Get highlighted map
map_image = await pipeline.get_highlighted_map_image()
```

### Demo Scripts

Run the emotion detection demo:

```bash
python demos/emotion_room_demo.py
```

This will:
1. Test all emotion classes with synthetic faces
2. Generate highlighted maps for each emotion
3. Save results to `demo_outputs/` directory

## ⚙️ Configuration

### Environment Variables

```bash
# Enable/disable emotion detection
ENABLE_EMOTION_DETECTION=true
ENABLE_ROOM_HIGHLIGHTING=true

# Emotion detection settings
EMOTION_DETECTION_FPS=10
FACE_DETECTION_METHOD=opencv  # opencv or mtcnn
EMONET_MODEL_PATH=pretrained/emonet_8.pth
N_EMOTION_CLASSES=8
```

### Configuration File

```python
# In config.py
enable_emotion_detection: bool = True
enable_room_highlighting: bool = True
emotion_detection_fps: int = 10
face_detection_method: str = "opencv"
emonet_model_path: str = "pretrained/emonet_8.pth"
n_emotion_classes: int = 8
```

## 📊 Performance

### Expected Performance

- **Face Detection**: 30+ FPS (OpenCV), 10-15 FPS (MTCNN)
- **Emotion Detection**: 10-15 FPS (EmoNet)
- **Room Highlighting**: Real-time map updates
- **Memory Usage**: ~500MB for EmoNet model
- **Latency**: <100ms per frame

### Optimization Tips

1. **Use OpenCV** for face detection for better performance
2. **Reduce emotion detection FPS** for lower CPU usage
3. **Use mock components** for development and testing
4. **Enable room highlighting** only when needed

## 🔧 Installation

### Dependencies

Install emotion detection dependencies:

```bash
pip install -r requirements_emonet.txt
```

### EmoNet Model

Download the EmoNet pretrained model:

```bash
mkdir -p pretrained
# Download emonet_8.pth to pretrained/ directory
```

### Face Alignment (Optional)

For advanced face detection with MTCNN:

```bash
pip install face-alignment
```

## 🎨 Customization

### Custom Emotion Mappings

```python
from emotion_room_mapper import EmotionRoomMapping

# Create custom mapping
custom_mapping = EmotionRoomMapping(
    emotion="excited",
    room_names=["Game Room", "Bar Room"],
    color=(255, 255, 0),  # Bright yellow
    intensity=1.0,
    animation_type="pulse",
    priority=4,
    conditions={"min_arousal": 0.7}
)

# Add to mapper
mapper.add_custom_mapping(custom_mapping)
```

### Custom Room Highlighting Rules

```python
from room_highlighting.room_highlighter import HighlightingRule

# Create custom rule
rule = HighlightingRule(
    emotion="happy",
    room_names=["Custom Room"],
    color=(0, 255, 0),
    intensity=0.8,
    animation_type="glow",
    priority=3
)

# Add to highlighter
room_highlighter.add_highlighting_rule(rule)
```

## 🐛 Troubleshooting

### Common Issues

1. **No faces detected**
   - Check if face detection method is correct
   - Ensure good lighting conditions
   - Try different face detection parameters

2. **Low emotion confidence**
   - Ensure face is clearly visible
   - Check if face is properly aligned
   - Verify EmoNet model is loaded correctly

3. **Room highlighting not working**
   - Check if map data is loaded
   - Verify emotion mapping rules
   - Ensure room names match exactly

4. **Performance issues**
   - Reduce emotion detection FPS
   - Use OpenCV instead of MTCNN
   - Enable mock components for testing

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('emotion_detection').setLevel(logging.DEBUG)
logging.getLogger('room_highlighting').setLevel(logging.DEBUG)
```

## 📈 Future Enhancements

### Planned Features

1. **Multi-person Support**: Detect emotions from multiple people
2. **Emotion History**: Track emotion trends over time
3. **Adaptive Learning**: Learn user preferences for room highlighting
4. **Voice Integration**: Combine with voice emotion detection
5. **Haptic Feedback**: Add tactile feedback for room highlighting
6. **Accessibility Features**: Enhanced support for visually impaired users

### Advanced Features

1. **Emotion Clustering**: Group similar emotions for better room selection
2. **Context Awareness**: Consider time of day, activity, and location
3. **Personalization**: User-specific emotion-to-room mappings
4. **Social Features**: Share emotion states with family members
5. **Health Monitoring**: Track emotional well-being over time

## 📚 References

- [EmoNet Paper](https://www.nature.com/articles/s42256-020-00280-0): "Estimation of continuous valence and arousal levels from faces in naturalistic conditions"
- [EmoNet Repository](https://github.com/face-analysis/emonet): Official implementation
- [OpenCV Face Detection](https://docs.opencv.org/4.x/d1/d5c/tutorial_py_face_detection.html)
- [MTCNN Face Detection](https://github.com/ipazc/mtcnn)

## 🤝 Contributing

To contribute to the emotion detection system:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

---

**Note**: This system is designed for the Mentra accessibility glasses project and integrates seamlessly with the existing SLAM and video processing pipeline. The emotion detection provides real-time feedback to help users navigate and interact with their environment based on their emotional state.
