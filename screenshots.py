import pyautogui
import os
import time

# This script was used multiple times, the following output_dir reflects the most recent usage
output_dir='annotated_images/phase-1/images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def capture_screenshots(output_dir=output_dir, interval=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    frame_count = 0
    while True:
        left = 0
        top = 36
        width = 615
        height = 921

        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        frame_path = os.path.join(output_dir, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}.png')
        screenshot.save(frame_path)
        frame_count += 1
        print(f"Captured {frame_path}")
        time.sleep(interval)

# Example usage
capture_screenshots(interval=.1)
