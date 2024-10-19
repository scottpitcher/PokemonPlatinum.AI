import pygetwindow as gw
import pyautogui
from PIL import Image
import os

print("Starting...")

def capture_active_window():
    # Bring DeSmuME to the front
    os.system("osascript -e 'tell application \"DeSmuME\" to activate'")

    # Get the active window
    window = gw.getActiveWindow()
    print(window)

    # Take a screenshot of the window's bounding box
    print("Capturing window")
    left, top, width, height = window.left, window.top, window.width, window.height
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    
    # Convert the screenshot to a Pillow image if needed for further processing
    image = Image.frombytes('RGB', screenshot.size, screenshot.tobytes())
    return image

# Example usage
screenshot_image = capture_active_window()
if screenshot_image:
    screenshot_image.show()  # Show the screenshot
    screenshot_image.save("desmume_screenshot.png")  # Save screenshot to file
