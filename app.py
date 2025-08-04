import cv2
import os
from PIL import Image
import numpy as np

# Open video
video_path = "assets/video/vid1.mov"
cap = cv2.VideoCapture(video_path)

# Load and resize replacement images
image_folder = "assets/original/"
compressed_folder = "assets/compressed/"
chunk_size = (16, 16)  # width, height
replacement_images = []
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).convert("RGBA")
        img_resized = img.resize(chunk_size, Image.Resampling.LANCZOS)
        # Save resized image to compressed folder as PNG
        compressed_path = os.path.join(
            compressed_folder, os.path.splitext(filename)[0] + ".png"
        )
        img_resized.save(compressed_path, format="PNG")
        replacement_images.append(np.array(img_resized))

# Check if video open
if not cap.isOpened():
    print("Error: Couldnt open video")
else:
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = "assets/video/output_pixelized.mp4"
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

num_images_used = 5  # Set this to the number of images you want to use (1-5)
# Map image names to priority (i1: darkest, ...)
image_priority = [f"i{i + 1}" for i in reversed(range(num_images_used))]
# Sort replacement_images by filename priority
sorted_images = [None] * num_images_used
for idx, filename in enumerate(os.listdir(image_folder)):
    name = os.path.splitext(filename)[0]
    if name in image_priority:
        sorted_images[image_priority.index(name)] = replacement_images[idx]

# Read and process all frames
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video
    out_frame = np.zeros_like(frame)
    # Dynamically calculate brightness boundaries for this frame
    chunk_brightness = []
    for y in range(0, height, chunk_size[1]):
        for x in range(0, width, chunk_size[0]):
            chunk = frame[y : y + chunk_size[1], x : x + chunk_size[0]]
            avg_brightness = np.mean(chunk[..., :3])
            chunk_brightness.append(avg_brightness)
    min_b = min(chunk_brightness)
    max_b = max(chunk_brightness)
    boundaries = np.linspace(min_b, max_b, num=num_images_used + 1)  # bins

    idx = 0
    for y in range(0, height, chunk_size[1]):
        for x in range(0, width, chunk_size[0]):
            avg_brightness = chunk_brightness[idx]
            # Find which bin this chunk falls into
            priority_idx = np.digitize(avg_brightness, boundaries) - 1
            priority_idx = int(np.clip(priority_idx, 0, num_images_used - 1))
            rep_img = sorted_images[priority_idx]
            if rep_img is not None:
                chunk = frame[y : y + chunk_size[1], x : x + chunk_size[0]]
                h, w = chunk.shape[:2]
                out_frame[y : y + h, x : x + w] = rep_img[:h, :w, :3]
            idx += 1
    out_video.write(out_frame)
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")
cap.release()
out_video.release()
print(f"Exported pixelized video to {out_path}")

# if __name__ == "__main__":
#     main()
