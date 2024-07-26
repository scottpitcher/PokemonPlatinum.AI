import os
from PIL import Image

images_in_path = 'train_data/label_training'
annotation_paths = ['pokemon_platinum_ai-9/train/labels', 'pokemon_platinum_ai-9/test/labels', 'pokemon_platinum_ai-9/valid/labels']


images_out_path = 'annotated_images/images'
annotation_out_path = 'annotated_images/labels'

images_and_annotations = []

if not os.path.exists(images_out_path):
    os.makedirs(images_out_path)
if not os.path.exists(annotation_out_path):
    os.makedirs(annotation_out_path)

# Gather images from image folder, in order oldest to newest to mantain sequence order
images_folder = sorted(os.listdir(images_in_path), key=lambda x: os.path.getmtime(os.path.join(images_in_path, x)))

for image_file in images_folder:
    annotation_name = f'frame_{images_folder.index(image_file)}.txt'

    # Skip DSStore file
    if image_file == ".DS_Store":
        print("Skipping DS_Store")
        continue

    image_path = os.path.join(images_in_path, image_file)
    image = Image.open(image_path)

    # Check if annotation already saved, if so, save the image to keep proper order and move on
    if os.path.exists(os.path.join(annotation_out_path, annotation_name)):
        images_and_annotations.append((image, None))
        print(f"Annotation for {image_file} already saved, continuing...")
        continue
    
    # Look for annotation file
    annotation_file = None
    image_basename = image_file.replace('.png', '_png')
    
    # Loop through all possible folders containing annotation
    for annotation_path in annotation_paths:
        # Get dir of current folder
        annotation_files = os.listdir(annotation_path)
        # Loop over all files in current folder
        for potential_annotation_file in annotation_files:
            # Check if the basename is in the filename
            if image_basename in potential_annotation_file:
                # If so, retrieve that filename
                annotation_file = os.path.join(annotation_path, potential_annotation_file)
                break
        # Ensure that once the file has been found, the above for loop stops and moves onto saving the image/annotation
        if annotation_file:
            break
    
    # If the annotation exists, save it, otherwise, save only the image for order purposes
    if annotation_file:
        images_and_annotations.append((image, annotation_file))
        print(f"Retrieved image {image_file} and its annotation successfully")
    else:
        images_and_annotations.append((image, None))
        print(f"No annotation found for image {image_file}, image sucessfully retrieved")


# write them all with standardised filenames
for i, (image, annotation_file) in enumerate(images_and_annotations):
    # Keeping all images saved with or w/out annotation for order purposes
    new_image_name = f'frame_{i}.png'
    new_annotation_name = f'frame_{i}.txt'


    # Check if the image is already saved, if not, save it, if so, check if it has an annotation
    if os.path.exists(os.path.join(images_out_path,new_image_name)) and os.path.exists(os.path.join(annotation_out_path,new_annotation_name)):
        print(f'Image and annotation for {new_image_name} already saved, continuing...')
        continue
    elif os.path.exists(os.path.join(images_out_path,new_image_name)) and annotation_file is None:
        print(f"Image {new_image_name} has no annotation created yet, continuing...")
        continue

    # If the annotation file exists it will save it, otherwise, it will alert that only the image was saved
    if annotation_file is not None:
        image.save(os.path.join(images_out_path, new_image_name))
        # Check if the annotation has already been saved, otherwise indicate
        if not os.path.exists(os.path.join(annotation_out_path,new_annotation_name)):
            with open(annotation_file, 'r') as af:
                annotation_content = af.read()
            with open(os.path.join(annotation_out_path, new_annotation_name), 'w') as af_out:
                af_out.write(annotation_content)
                
            print(f"Image {new_image_name} and its annotation saved")

print("Completed")

images_out_folder = sorted(os.listdir(images_out_path))

j=0
# Check that annotations match up
for i in range(len(images_folder)):
    new_name = f'frame_{i}.txt'
    new_path = os.path.join(annotation_out_path, new_name)
    
    # counting files that did not have annotations
    if os.path.exists(new_path):
        with open(new_path, 'r') as af:
            new_content = af.read()
    else:
        new_content = "None"
    
    base_name = images_folder[i].replace('.png','_png')
    if base_name =='.DS_Store':
        continue
    original_content = "None"
    annotation_file = None
    # Loop through all possible folders containing annotation
    for annotation_path in annotation_paths:
        # Get dir of current folder
        annotation_files = os.listdir(annotation_path)
        # Loop over all files in current folder
        for potential_annotation_file in annotation_files:
            # Check if the basename is in the filename
            if base_name in potential_annotation_file:
                # If so, retrieve that filename
                annotation_file = os.path.join(annotation_path, potential_annotation_file)
                break
        # Ensure that once the file has been found, the above for loop stops and moves onto saving the image/annotation
        if annotation_file:
            break
    
    if annotation_file is not None:
        with open(annotation_file, 'r') as af:
            original_content = af.read()

    if original_content == new_content:
        j+=1
    else:
        print(f"{base_name} has a mismatch with {new_name}")

    if (j+1)==len(images_out_folder):
        print("All annotations were processed correctly!")
