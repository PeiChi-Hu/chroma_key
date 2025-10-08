🎬 Chroma Key Video Compositor

This project is a simple Python tool for greenscreen video compositing.
It lets you replace a green background in a video with a custom image, using OpenCV for interactive color sampling and mask adjustment.

✨ Features

Interactive ROI (drag on the video to select the green patch to key out).

Adjustable tolerance and softness via trackbars for fine control.

Real-time preview of mask and composited output.

Saves the processed video with the chosen background.

🛠 Requirements

Python 3.7+

OpenCV
 (cv2)

NumPy

Install dependencies:

pip install opencv-python numpy

📂 File Overview

submission2.py → Main script for chroma keying.

Input video: greenscreen-asteroid.mp4 (replace with your own if needed).

Background image: party.jpg (replace with your own background).

Output video: output_video.mp4.

You can change file names at the top of the script:

input_video = "greenscreen-asteroid.mp4"
background  = "party.jpg"
output_video = "output_video.mp4"

▶️ Usage

Run the script:

python submission2.py


The first frame of your video opens in a window.

Drag a rectangle with the mouse over the green area to sample the key color.

Use the trackbars:

Tolerance → adjusts how strictly green is detected.

Softness → controls edge smoothing.

Watch the live preview (bottom-right corner shows current mask).

Press S to save and exit.

💾 Output

The composited video will be written to output_video.mp4.

Background will be automatically resized to match the video dimensions.
