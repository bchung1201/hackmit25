"""
Simple webcam test to verify video capture works
"""

import cv2
import sys

def test_webcam():
    """Test webcam capture"""
    print("🎥 Testing webcam capture...")
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        print("Make sure your webcam is connected and not being used by another application")
        return False
    
    print("✅ Webcam opened successfully")
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📐 Resolution: {width}x{height}")
    print(f"📊 FPS: {fps}")
    
    print("\n🎬 Starting video capture...")
    print("Press 'q' to quit, 's' to save a frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Add frame counter overlay
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Webcam Test', frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("👋 Quitting...")
            break
        elif key == ord('s'):
            filename = f"captured_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Frame saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Test completed. Processed {frame_count} frames")
    return True

if __name__ == "__main__":
    try:
        success = test_webcam()
        if success:
            print("\n🎉 Webcam test successful!")
            print("You can now run the real video processing demo:")
            print("python3 real_video_demo.py")
        else:
            print("\n❌ Webcam test failed")
            print("Please check your webcam connection and try again")
    except KeyboardInterrupt:
        print("\n👋 Test interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
