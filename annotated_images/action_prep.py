import os
import json
from PIL import Image
from natsort import natsorted

# Define directories
images_dir = 'annotated_images/phase-1/images'
actions_dir = 'annotated_images/phase-1/actions'

def get_action_from_user():
    """Function to receive an action from the user based on 9 choices"""
    action = ''
    # Error checking to ensure that choice is within the list
    while action not in ["up", "down", "left", "right", "a", "b", "x", "y", "none"]:
        action = input("Enter the action for this image (up/down/left/right/a/b/x/y/none): ")
    return {"action": action}

def annotate_images(images_dir, actions_dir):
    if not os.path.exists(actions_dir):
        os.makedirs(actions_dir)

    images = natsorted(os.listdir(images_dir))  # Using this sort function to ensure proper order
    prev_idx = 0  # Initializing the prev_idx

    # Loop through every image idx
    for idx, image_name in enumerate(images):

        # Get current image path
        image_path = os.path.join(images_dir, image_name)

        # Skip the DS Store file for error checking
        if image_name == '.DS_Store':
            continue

        # Grab current image to view for user
        image = Image.open(image_path)

        # Replace extension for new filename for the action
        action_file = images[idx].replace('.jpg', '.json')
        action_file_path = os.path.join(actions_dir, action_file)

        # Check if the action file exists and if it is emptyna
        if os.path.exists(action_file_path):
            # Check if the file is empty
            with open(action_file_path, 'r') as f:
                content = f.read().strip()  # Strip any whitespace characters
                # Check if the file is empty or contains only whitespace
                if not content:
                    print(f"{action_file} is empty.")
                else:
                    print(f"{action_file} is not empty.")
                    continue
        
        image.show()
        action = get_action_from_user()
        image.close()

        with open(action_file_path, 'w') as f:
            json.dump(action, f)

# Run the annotation
annotate_images(images_dir, actions_dir)