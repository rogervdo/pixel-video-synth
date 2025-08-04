import cv2
import os
from PIL import Image
import numpy as np


def load_and_resize_images(image_folder, compressed_folder, chunk_size):
    replacement_images = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert("RGBA")
            img_resized = img.resize(chunk_size, Image.Resampling.LANCZOS)
            compressed_path = os.path.join(
                compressed_folder, os.path.splitext(filename)[0] + ".png"
            )
            img_resized.save(compressed_path, format="PNG")
            replacement_images.append(np.array(img_resized))
    return replacement_images


def sort_images_by_priority(image_folder, replacement_images, num_images_used):
    image_priority = [f"i{i + 1}" for i in reversed(range(num_images_used))]
    sorted_images = [None] * num_images_used
    for idx, filename in enumerate(os.listdir(image_folder)):
        name = os.path.splitext(filename)[0]
        if name in image_priority:
            sorted_images[image_priority.index(name)] = replacement_images[idx]
    return sorted_images


def pixelize_video(
    video_path,
    out_path,
    image_folder,
    compressed_folder,
    chunk_size,
    num_images_used,
    alpha_mode,
):
    """
    alpha_mode: "static" uses 255 as max brightness for consistent boundaries,
                "dynamic" uses the actual max brightness found in each frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldnt open video")
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    replacement_images = load_and_resize_images(
        image_folder, compressed_folder, chunk_size
    )
    sorted_images = sort_images_by_priority(
        image_folder, replacement_images, num_images_used
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video
        out_frame = np.zeros_like(frame)
        chunk_brightness = []
        for y in range(0, height, chunk_size[1]):
            for x in range(0, width, chunk_size[0]):
                chunk = frame[y : y + chunk_size[1], x : x + chunk_size[0]]
                avg_brightness = np.mean(chunk[..., :3])
                chunk_brightness.append(avg_brightness)
        min_b = min(chunk_brightness)

        if alpha_mode == "dynamic":
            max_b = max(chunk_brightness)
        else:  # static mode
            max_b = 255

        boundaries = np.linspace(min_b, max_b, num=num_images_used + 1)

        idx = 0
        for y in range(0, height, chunk_size[1]):
            for x in range(0, width, chunk_size[0]):
                avg_brightness = chunk_brightness[idx]
                priority_idx = np.digitize(avg_brightness, boundaries) - 1
                priority_idx = int(np.clip(priority_idx, 0, num_images_used - 1))
                rep_img = sorted_images[priority_idx]
                if rep_img is not None:
                    chunk = frame[y : y + chunk_size[1], x : x + chunk_size[0]]
                    h, w = chunk.shape[:2]
                    # Handle alpha channel and blend
                    if rep_img.shape[2] == 4:
                        rgb = rep_img[:h, :w, :3].astype(np.float32)
                        alpha = rep_img[:h, :w, 3].astype(np.float32)
                        chunk_rgb = chunk[:h, :w, :3].astype(np.float32)

                        if alpha_mode == "dynamic":
                            # Use the actual alpha channel per pixel, normalized to [0,1]
                            alpha_norm = alpha / 255.0
                        else:
                            # Use static alpha value (255) for all pixels
                            alpha_norm = np.ones_like(alpha)
                        chunk_rgb = chunk[:h, :w, :3].astype(np.float32)
                        # Blend: output = alpha * replacement + (1 - alpha) * original
                        blended = (
                            alpha_norm[..., None] * rgb
                            + (1 - alpha_norm[..., None]) * chunk_rgb
                        )

                        out_frame[y : y + h, x : x + w] = blended.clip(0, 255).astype(
                            np.uint8
                        )
                    else:
                        out_frame[y : y + h, x : x + w] = rep_img[:h, :w, :3]
                idx += 1
        out_video.write(out_frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    cap.release()
    out_video.release()
    print(f"Exported pixelized video to {out_path}")


def main():
    video_path = "assets/video/vid1.mov"
    out_path = "assets/video/output_pixelized.mp4"
    image_folder = "assets/original/"
    compressed_folder = "assets/compressed/"
    chunk_size = (16, 16)
    num_images_used = 5
    # Set alpha_mode to "static" or "dynamic"
    alpha_mode = "static"  # or "dynamic"
    pixelize_video(
        video_path,
        out_path,
        image_folder,
        compressed_folder,
        chunk_size,
        num_images_used,
        alpha_mode,
    )


if __name__ == "__main__":
    main()
