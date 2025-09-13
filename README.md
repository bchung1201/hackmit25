# Mentra Video Processing Pipeline

A focused real-time video processing system that combines Mentra glasses, Gaussian Splatting, SAM segmentation, and Claude VLM to create intelligent 3D environment understanding.

## 🎯 Project Overview

This pipeline represents a cutting-edge approach to real-time 3D environment understanding, designed specifically for the HackMIT 2025 hackathon. It focuses on the core video processing workflow: streaming, segmentation, 3D reconstruction, and AI analysis.

### Key Features

- **Real-time Video Streaming**: Capture first-person perspective from Mentra glasses
- **3D Scene Reconstruction**: Use Modal compute for Gaussian Splatting reconstruction
- **Object Segmentation**: SAM-powered real-time object detection and segmentation
- **AI Understanding**: Claude VLM for scene analysis and environment understanding
- **Performance Optimization**: Focused on real-time processing with minimal latency
- **Accessibility Focus**: Built-in accessibility assessment capabilities

## 🏗️ Architecture

The pipeline consists of five main components working in harmony:

```
Mentra Glasses → Video Streaming → 3D Reconstruction → Segmentation → VLM Analysis
     ↓                                                                    ↑
Voice Commands ← Wispr Integration ← Voice Response ← Command Processing ←
```

### Component Details

1. **Video Streaming** (`video_streaming.py`)
   - Real-time capture from Mentra glasses
   - RTSP streaming support
   - Frame buffering and callbacks

2. **3D Reconstruction** (`modal_3d_reconstruction.py`)
   - Modal-based cloud compute
   - Gaussian Splatting for real-time reconstruction
   - GPU-optimized processing

3. **Segmentation** (`sam_segmentation.py`)
   - SAM (Segment Anything Model) integration
   - Real-time object detection
   - Trajectory tracking

4. **VLM Processing** (`claude_vlm.py`)
   - Claude VLM for scene understanding
   - Natural language interaction
   - Accessibility assessment

5. **Voice Integration** (`wispr_voice.py`)
   - Speech-to-text with Whisper
   - Voice command processing
   - Hands-free operation

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for SAM)
- Mentra glasses (for production)
- API keys for Claude and Modal

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hackmit25
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export CLAUDE_API_KEY="your-claude-api-key"
export MODAL_TOKEN="your-modal-token"
```

### Running the Pipeline

#### Development Mode (with Mock Components)
```bash
python main.py
```

#### Production Mode (with Real Components)
```bash
python pipeline_integration.py
```

## 🎮 Usage Examples

### Basic Scene Analysis
```python
from pipeline_integration import MentraRealityPipeline

# Initialize pipeline
pipeline = MentraRealityPipeline(config, use_mock_components=False)

# Start pipeline
await pipeline.start()

# Ask questions about the scene
response = await pipeline.handle_voice_command("Describe this room to me")
print(response)
```

### Voice Commands
- "Describe this room to me"
- "Where is the nearest exit?"
- "Count the objects in this scene"
- "What accessibility features do you see?"
- "Help me navigate to the kitchen"

### Programmatic Interaction
```python
# Get current scene description
scene = pipeline.state.current_scene_description
print(f"Room type: {scene.room_type}")
print(f"Objects found: {len(scene.objects)}")

# Check accessibility
accessibility = await pipeline.vlm_processor.assess_accessibility(frame)
print(f"Wheelchair accessible: {accessibility['wheelchair_accessible']}")
```

## 🏆 Hackathon Alignment

### Sponsor Track Coverage

✅ **Mentra**: Best Use of MentraOS
- Real-time video streaming from glasses
- Native OS integration for seamless data flow

✅ **YC Challenge**: Challenging Matterport
- Dynamic, AI-powered digital twins vs. static scans
- Real-time understanding vs. post-processing analysis

✅ **Anthropic**: Advanced Claude Integration
- Multi-modal VLM processing (video + 3D + segmentation)
- Complex scene understanding and natural language interaction

✅ **Modal**: Optimal Cloud Compute Usage
- GPU-intensive Gaussian Splatting workloads
- Scalable, burstable compute for real-time processing

✅ **Wispr**: Yap ur App Integration
- Voice-first interaction model
- Hands-free operation perfect for wearable devices

### Judging Criteria

- **Innovation (30%)**: Real-time fusion of cutting-edge technologies
- **Technical Complexity (30%)**: Multi-modal AI pipeline with latency optimization
- **Impact (30%)**: Accessibility, emergency response, real estate applications

## 🔧 Configuration

### Pipeline Configuration
```python
config = {
    "mentra_stream_url": "rtsp://mentra-glasses.local:8554/stream",
    "video_buffer_size": 10,
    "modal_app_name": "mentra-reality-pipeline",
    "sam_model_type": "vit_h",
    "claude_api_key": "your-api-key",
    "claude_model": "claude-3-5-sonnet-20241022",
    "whisper_model_size": "base",
    "language": "en"
}
```

### Performance Tuning
- Adjust `video_buffer_size` for memory usage
- Modify `sam_model_type` for speed vs. accuracy tradeoff
- Configure `max_queue_size` for processing latency

## 🚧 Development Status

- [x] Project structure and architecture
- [x] Video streaming framework
- [x] Modal 3D reconstruction setup
- [x] SAM segmentation integration
- [x] Claude VLM processing
- [x] Wispr voice integration
- [x] Pipeline orchestration
- [ ] Latency optimization
- [ ] Production deployment
- [ ] Testing and validation

## 🤝 Contributing

This project is developed for HackMIT 2025. Contributions are welcome for:
- Performance optimizations
- Additional voice commands
- Enhanced accessibility features
- Integration improvements

## 📄 License

This project is developed for HackMIT 2025 hackathon purposes.

## 🙏 Acknowledgments

- Mentra for the innovative smart glasses platform
- Modal for scalable cloud compute infrastructure
- Anthropic for advanced VLM capabilities
- Meta for the SAM segmentation model
- OpenAI for Whisper speech recognition
- The open-source community for foundational tools

---

**Built for HackMIT 2025** 🚀
