import os
import cv2
import yt_dlp as youtube_dl

youtube_urls = ["",
                ]

def download_video(urls):
    """Downloading videos from YouTube of gameplay"""
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join('train_data/videos/', '%(title)s.%(ext)s'),
    }
    for url in urls:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

def get_relative_file_paths(directory):
    relative_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), directory)
            relative_paths.append(relative_path)
    return relative_paths

def extract_frames(video_paths, output_dir="train_data/frames", frame_rate=1):
    """Extract the frame data from each video"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_dir, video_name)
        
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)
        
        absolute_video_path = os.path.join('train_data/videos', video_path)
        cap = cv2.VideoCapture(absolute_video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {absolute_video_path}")
            continue
        
        frame_count = 0
        success, frame = cap.read()
        while success:
            if frame_count % frame_rate == 0:
                frame_filename = os.path.join(video_output_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
            success, frame = cap.read()
            frame_count += 1
        
        cap.release()
        print(f"Extracted {frame_count} frames from {absolute_video_path}")

if __name__ == "__main__":
    download_video(youtube_urls)
    video_files = get_relative_file_paths("train_data/videos")
    extract_frames(video_paths=video_files)
