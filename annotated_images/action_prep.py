import os
import json
from PIL import Image

def get_action_from_user():
    action = input("Enter the action for the previous image: ")
    return {"action": action}

def annotate_images(images_dir, actions_dir):
    if not os.path.exists(actions_dir):
        os.makedirs(actions_dir)
    
    images = sorted(os.listdir(images_dir))
    
    for idx, image_name in enumerate(images):
        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path)
        image.show()

        if idx > 0:
            action = get_action_from_user()
            action_file = os.path.join(actions_dir, images[idx-1].replace('.png', '.json'))
            with open(action_file, 'w') as f:
                json.dump(action, f)
        
        input("Press Enter to continue to the next image...")

    # For the last image
    action = get_action_from_user()
    action_file = os.path.join(actions_dir, images[-1].replace('.png', '.json'))
    with open(action_file, 'w') as f:
        json.dump(action, f)

images_dir = 'path_to_images'  # Replace with the path to your images directory
actions_dir = 'path_to_actions'  # Replace with the path to your actions directory
annotate_images(images_dir, actions_dir)
