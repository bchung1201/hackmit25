# Setting Up Real Video Processing

## 🎥 Current Status

**The pipeline is currently using MOCK data, not real video!**

The demos you've seen are simulating video processing with fake data. Here's how to set up real video processing.

## 🛠️ Quick Setup

### Step 1: Install Dependencies
```bash
# Install OpenCV for video processing
pip install opencv-python

# Install other dependencies
pip install numpy torch anthropic modal
```

### Step 2: Test Your Webcam
```bash
python3 simple_webcam_test.py
```

### Step 3: Run Real Video Demo
```bash
python3 real_video_demo.py
```

## 🎬 What You'll See

### Mock Processing (Current)
- **Frame**: `'mock_frame_1'` (just a string)
- **Objects**: Pre-defined list (chair, table, laptop)
- **Scene**: Hardcoded "office" response
- **3D Model**: Simulated gaussian count

### Real Processing (After Setup)
- **Frame**: Actual pixel data from your webcam
- **Objects**: SAM model detecting real objects in the scene
- **Scene**: Claude VLM analyzing what it actually sees
- **3D Model**: Gaussian Splatting on real video frames

## 🔧 Configuration

### For Real Video Processing
```python
# In config.py or main.py
config = {
    "use_mock_components": False,  # Set to False for real processing
    "mentra_stream_url": "rtsp://mentra-glasses.local:8554/stream",  # For Mentra glasses
    # Or use webcam: cv2.VideoCapture(0)
}
```

### API Keys (Optional)
```bash
export CLAUDE_API_KEY="your-claude-api-key"
export MODAL_TOKEN="your-modal-token"
```

## 🎯 Video Sources

### 1. Webcam (Easiest)
```python
cap = cv2.VideoCapture(0)  # Default webcam
```

### 2. Video File
```python
cap = cv2.VideoCapture("path/to/video.mp4")
```

### 3. Mentra Glasses (Production)
```python
cap = cv2.VideoCapture("rtsp://mentra-glasses.local:8554/stream")
```

## 🚀 Testing Steps

1. **Install OpenCV**: `pip install opencv-python`
2. **Test webcam**: `python3 simple_webcam_test.py`
3. **Run real demo**: `python3 real_video_demo.py`
4. **Configure pipeline**: Set `use_mock_components=False`
5. **Run pipeline**: `python3 main.py`

## 🎭 Mock vs Real Comparison

| Aspect | Mock (Current) | Real (After Setup) |
|--------|----------------|-------------------|
| Video Source | Simulated frames | Webcam/video file |
| Object Detection | Hardcoded objects | SAM model |
| Scene Analysis | Fake responses | Claude VLM |
| 3D Reconstruction | Simulated gaussians | Gaussian Splatting |
| Performance | Instant | Real processing time |

## 🎉 Expected Results

After setup, you'll see:
- **Live video feed** from your webcam
- **Real object detection** with bounding boxes
- **Actual scene analysis** from Claude VLM
- **Real 3D reconstruction** with Gaussian Splatting
- **Performance metrics** from actual processing

## 🆘 Troubleshooting

### Webcam Issues
- Make sure webcam is not being used by another app
- Try different camera indices: `cv2.VideoCapture(1)`, `cv2.VideoCapture(2)`
- Check camera permissions

### Dependencies Issues
- Install OpenCV: `pip install opencv-python`
- Install PyTorch: `pip install torch torchvision`
- Install other deps: `pip install -r requirements.txt`

### API Issues
- Get Claude API key from Anthropic
- Get Modal token from Modal Labs
- Set environment variables

---

**Once you complete the setup, you'll have REAL video processing instead of mock data!** 🚀
