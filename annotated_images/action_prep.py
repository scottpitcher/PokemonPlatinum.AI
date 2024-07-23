import os
import json
from PIL import Image
from natsort import natsorted

def get_action_from_user():
    action = ''
    while action not in ["up","down","left","right","a","b","x","y","none"]:
        action = input("Enter the action for the previous image (up/down/left/right/a/b/x/y/none): ")
    return {"action": action}

def annotate_images(images_dir, actions_dir):
    if not os.path.exists(actions_dir):
        os.makedirs(actions_dir)
    
    images = natsorted(os.listdir(images_dir)) # Using this sort function to ensure proper order
    prev_idx = 0                               # Initialising the prev_idx
    # Loop through every image idx
    for idx, image_name in enumerate(images):

        # Get current image path
        image_path = os.path.join(images_dir, image_name)

        # skip the DS Store file for error checking
        if image_name =='.DS_Store':
            continue

        # Grab current image to view for user
        image = Image.open(image_path)

        # Based on prev and current image, decide on action taken for prev; replace extension for new filename
        action_file = images[idx].replace('.png', '.json')
        # Save the current idx to be used for the next iteration
        # prev_idx = idx

        # Skip files that have already been processed
        if os.path.exists(os.path.join(actions_dir,action_file)):
            print(f"{action_file} has already been processed, continuing")
            continue
        else:
            image.show()

        if idx >= 0:
            action = get_action_from_user()
            image.close()
            action_file = os.path.join(actions_dir, action_file)
            with open(action_file, 'w') as f:
                json.dump(action, f)
        
        
        
    # For the last image
    action = get_action_from_user()
    action_file = os.path.join(actions_dir, images[-1].replace('.png', '.json'))
    with open(action_file, 'w') as f:
        json.dump(action, f)
        

images_dir = 'annotated_images/images'
actions_dir = 'annotated_images/actions'
annotate_images(images_dir, actions_dir)


