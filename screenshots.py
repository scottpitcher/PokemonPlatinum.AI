import pyautogui
import os
import time


def capture_screenshots(output_dir='train_data/label_training', interval=3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    frame_count = 0
    while True:
        left = 539
        top = 87
        width = 391
        height = 581

        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        frame_path = os.path.join(output_dir, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}.png')
        screenshot.save(frame_path)
        frame_count += 1
        print(f"Captured {frame_path}")
        time.sleep(interval)

# Example usage
capture_screenshots(interval=10)

