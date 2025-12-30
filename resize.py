import os
from PIL import Image

def resize_images_in_folder(folder_path, target_size=(48, 48)):
    # Loop through each class folder inside the train/val folder
    for class_folder in os.listdir(folder_path):
        class_folder_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_folder_path):  # Ensure it's a directory
            # Loop through each image in the class folder
            for image_name in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_name)
                if image_path.endswith(('jpg', 'jpeg', 'png')):  # Only process image files
                    try:
                        # Open image, resize and save it
                        with Image.open(image_path) as img:
                            img_resized = img.resize(target_size)
                            img_resized.save(image_path)  # Save the resized image
                        print(f"Resized {image_name} in {class_folder}")
                    except Exception as e:
                        print(f"Error processing {image_name}: {e}")

# Paths to train and validation folders (change these to your local paths)
train_folder = 'F:/6thSem/SignLanguage/data_set/splitdataset128x128/train'
val_folder = 'F:/6thSem/SignLanguage/data_set/splitdataset128x128/val'

# Resize images in both train and val folders
resize_images_in_folder(train_folder)
resize_images_in_folder(val_folder)

print("Resizing completed!")