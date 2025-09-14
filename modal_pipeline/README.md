Modal 3D Reconstruction Pipeline
This project demonstrates a 3D reconstruction pipeline running on the Modal platform. It processes an input video file, performs dense depth estimation with MiDaS, simulates a SLAM trajectory, and fuses the results into a 3D point cloud.

The final output is saved in two formats to a persistent Modal Shared Volume:

A sequence of raw 3D point cloud files (.ply).

A rendered MP4 video showing the reconstruction process.

File Structure
/
|
├── app.py             # The main Modal script with all the logic.
|
└── desk_sequence.mp4  # The example video file.

Prerequisites
A Modal Account: Sign up at modal.com and install the CLI.

Input Video: You need a video file named desk_sequence.mp4 in the same directory as app.py. You can get the recommended video for this demo by:

Downloading the rgb.tgz file from the TUM RGB-D freiburg1_desk dataset.

Extracting the .png images from it.

Converting the images to an MP4 video using FFmpeg:

ffmpeg -framerate 30 -pattern_type glob -i 'rgb/*.png' -c:v libx264 -pix_fmt yuv420p desk_sequence.mp4

How to Run
Deploy the App:
Open your terminal in this directory and deploy the Modal application. This only needs to be done once or when you change the code.

modal deploy app.py

Run the Pipeline:
Execute the main function from your local machine. This will trigger the functions to run remotely on Modal's infrastructure.

modal run app.py

This command will first run the lengthy process_video_to_files function, followed by the create_video_from_frames function. Expect the first step to take several minutes.

Accessing the Output
The output files are saved to a Modal Shared Volume named reconstruction-output. After the run completes, you can download the files to your local machine using the Modal CLI.

To download the final MP4 video:

modal volume get reconstruction-output reconstruction_video.mp4 .

To download the entire folder of raw .ply files:

modal volume get reconstruction-output raw_pcd .

You can now view reconstruction_video.mp4 with any media player or inspect the .ply files with 3D software like MeshLab.