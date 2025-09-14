import modal
import cv2
import torch
import open3d as o3d
import numpy as np
import os
import subprocess
import time

# --- Configuration ---
# This points to the video file in the same directory.
# Modal automatically includes this file when we run the app.
VIDEO_FILE_PATH = "./desk_sequence.mp4" 

# --- Camera Intrinsics ---
# These are standard values for the TUM RGB-D dataset 'freiburg1_desk' sequence.
# For other videos, these would need to be changed.
IMG_HEIGHT = 480
IMG_WIDTH = 640
FX = 525.0
FY = 525.0
CX = 319.5
CY = 239.5

# --- Modal Setup ---
# Create a persistent SharedVolume to store our output files.
# You can see and manage this in the Modal UI under the "Volumes" tab.
volume = modal.SharedVolume(name="reconstruction-output")
stub = modal.Stub("mentra-3d-reconstruction-saver")

# Define the container image with all necessary dependencies.
# This includes PyTorch with CUDA support and FFmpeg for video creation.
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", "torchvision", "opencv-python-headless", "open3d", "numpy", "timm"
).run_commands(
    "pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch torchvision --upgrade",
    "apt-get update && apt-get install -y ffmpeg"
)

# A class to manage the state and logic of the 3D reconstruction.
class ReconstructionEngine:
    def __init__(self, output_dir="/output"):
        """
        Initializes all models, the global point cloud, and the headless renderer.
        """
        print("Initializing Reconstruction Engine...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        
        # Create subdirectories in our shared volume for organized output
        self.raw_pcd_dir = os.path.join(output_dir, "raw_pcd")
        self.render_dir = os.path.join(output_dir, "renders")
        os.makedirs(self.raw_pcd_dir, exist_ok=True)
        os.makedirs(self.render_dir, exist_ok=True)

        # 1. Load Depth Estimation Model (MiDaS)
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.midas_transform = midas_transforms.small_transform
        
        # 2. Initialize SLAM System (Simulated for this demo)
        self.frame_count = 0
        
        # 3. Initialize Global Point Cloud that will accumulate points
        self.global_pcd = o3d.geometry.PointCloud()
        
        # 4. Initialize a headless Open3D renderer for creating video frames
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=1280, height=720, visible=False) # Headless
        self.vis.add_geometry(self.global_pcd)
        print("Engine Initialized.")

    def process_frame(self, frame_bgr):
        """
        Processes a single video frame, updates the 3D map, and saves outputs.
        """
        self.frame_count += 1
        
        # A. Get Camera Pose (Simulated for this demo)
        # A real SLAM system would compute this pose.
        pose = np.eye(4)
        pose[2, 3] = self.frame_count * -0.05 # Simple backward motion

        # B. Get Depth Map from MiDaS
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self.midas_transform(frame_rgb).to(self.device)
        with torch.no_grad():
            prediction = self.midas(image_tensor)
            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=frame_rgb.shape[:2], mode="bicubic", align_corners=False).squeeze()
        depth_map = prediction.cpu().numpy()

        # C. Create Point Cloud for the current frame and fuse into the global map
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(frame_rgb), o3d.geometry.Image(depth_map),
            depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        intrinsics = o3d.camera.PinholeCameraIntrinsic(IMG_WIDTH, IMG_HEIGHT, FX, FY, CX, CY)
        frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        frame_pcd.transform(pose)
        
        self.global_pcd += frame_pcd
        self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=0.03) # Critical for performance

        # D. Save the outputs to the shared volume
        # Option 1: Save the raw 3D point cloud file (.ply format)
        pcd_path = os.path.join(self.raw_pcd_dir, f"pcd_{self.frame_count:05d}.ply")
        o3d.io.write_point_cloud(pcd_path, self.global_pcd)

        # Option 2: Render an image of the current point cloud for the final video
        self.vis.update_geometry(self.global_pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        render_path = os.path.join(self.render_dir, f"frame_{self.frame_count:05d}.png")
        self.vis.capture_screen_image(render_path, do_render=True)
        
        print(f"Processed and saved frame {self.frame_count}. Global points: {len(self.global_pcd.points)}")

    def cleanup(self):
        self.vis.destroy_window()

@stub.function(
    image=image,
    gpu="A10G", # A capable GPU is needed for this task
    shared_volumes={"/output": volume}, # Mount the volume at the /output path in the container
    timeout=1800 # Allow up to 30 minutes for the job to run
)
def process_video_to_files():
    """
    This function runs the main reconstruction loop over the entire video.
    """
    engine = ReconstructionEngine()
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        engine.process_frame(frame_resized)
    
    cap.release()
    engine.cleanup()
    end_time = time.time()
    
    print(f"Finished processing video. Total time: {end_time - start_time:.2f} seconds.")
    # Commit the changes to the SharedVolume to ensure they are saved permanently.
    volume.commit()

@stub.function(
    image=image,
    shared_volumes={"/output": volume},
    timeout=600
)
def create_video_from_frames():
    """
    This function uses FFmpeg to stitch the rendered PNG frames into a single MP4 video.
    """
    print("🎬 Starting video creation from rendered frames...")
    
    input_path = "/output/renders/frame_%05d.png"
    output_path = "/output/reconstruction_video.mp4"
    
    ffmpeg_command = [
        "ffmpeg",
        "-y", # Overwrite output file if it exists
        "-framerate", "30",
        "-i", input_path,
        "-c:v", "libx24",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    
    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"✅ Video created and saved to {output_path}")
        volume.commit()
    except subprocess.CalledProcessError as e:
        print("FFmpeg command failed.")
        print("FFmpeg stdout:", e.stdout)
        print("FFmpeg stderr:", e.stderr)


@stub.local_entrypoint()
def main():
    """
    This is the main entrypoint you run from your command line.
    It orchestrates the two main steps of the pipeline.
    """
    print("--- Step 1: Processing video and generating raw files. This will take a while... ---")
    process_video_to_files.remote()
    
    print("\n--- Step 2: Stitching rendered frames into a final MP4 video. ---")
    create_video_from_frames.remote()
    
    print("\n All steps complete! You can now download the outputs from the 'reconstruction-output' volume.")
