import os
import cv2
import shutil
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def extract_ball_objects(image_folder, label_folder, ball_class=0):
    """
    Extracts small object crops (balls) from dataset.
    :param image_folder: Path to images
    :param label_folder: Path to YOLO labels
    :param ball_class: Class index of the ball
    :return: List of (cropped ball image, bounding box)
    """
    ball_objects = []

    for img_file in os.listdir(image_folder):
        if not img_file.endswith(".jpg"):
            continue

        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(label_folder, img_file.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x, y, bw, bh = map(float, line.strip().split())

                if int(cls) == ball_class:
                    # Convert YOLO format to pixel coordinates
                    x1 = int((x - bw / 2) * w)
                    y1 = int((y - bh / 2) * h)
                    x2 = int((x + bw / 2) * w)
                    y2 = int((y + bh / 2) * h)

                    ball_crop = img[y1:y2, x1:x2]  # Crop the ball
                    if ball_crop.size > 0:
                        ball_objects.append((ball_crop, (x, y, bw, bh)))

    return ball_objects


def copy_paste_augmentation(img, labels, objects, min_balls=3, max_balls=6):
    """
    Apply copy-paste augmentation by inserting extracted objects into a new image.
    :param img: Original image (numpy array)
    :param labels: Labels for the image (YOLO format)
    :param objects: List of extracted small objects (e.g., footballs)
    :param paste_prob: Probability of applying augmentation
    :return: Augmented image and updated labels
    """
    
    # if augment first save
    h, w, _ = img.shape
    new_labels = labels.copy()

    # avoid modifying original image and labels
    img = img.copy()
    new_labels = labels.copy()

    # Randomly select number of objects to paste
    num_objects = random.randint(min_balls, max_balls)
    objs_to_copy = random.sample(objects, num_objects)

    for obj in objs_to_copy:
        
        obj_img, _ = obj

        # Get a random position in the image
        x_center = random.randint(int(0.25*w), int(0.66*w) - obj_img.shape[1])
        y_center = random.randint(int(0.25*h), int(0.66*h) - obj_img.shape[0])

        # Paste the object onto the image
        img[y_center:y_center+obj_img.shape[0], x_center:x_center+obj_img.shape[1]] = obj_img

        # Convert new object position into YOLO format
        x_norm = (x_center + obj_img.shape[1] / 2) / w
        y_norm = (y_center + obj_img.shape[0] / 2) / h
        w_norm = obj_img.shape[1] / w
        h_norm = obj_img.shape[0] / h

        # Append new object label
        new_labels.append([0, x_norm, y_norm, w_norm, h_norm])

    return img, new_labels

import matplotlib.pyplot as plt
import cv2
import numpy as np

def visualize_batch(dataset, num_images=4):
    """
    Visualizes a batch of training images with bounding boxes.
    
    :param dataset: CustomYOLODataset object
    :param num_images: Number of images to display
    """
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for i in range(num_images):
        # Get image and labels
        img, labels = dataset[i]

        # Reverse normalization
        img = unnormalize(img)

        # Draw bounding boxes
        h, w, _ = img.shape
        for label in labels:
            class_id, x_center, y_center, box_w, box_h = label

            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            # Set color (Green for balls, Red for other objects)
            color = (0, 255, 0) if int(class_id) == 0 else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Show image
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        axes[i].axis("off")

    plt.show()

def unnormalize(img, mean=[0.5], std=[0.5]):
    """ Reverses normalization for visualization """
    img = img.numpy().transpose(1, 2, 0).copy()  # Ensure copy to avoid modifying original
    img = img * np.array(std) + np.array(mean)  # Undo normalization
    img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Scale and convert to uint8
    return img


import os
import cv2
import numpy as np

def bbox_copy_paste(root_folder, dataset_destination_path, num_copies, overwrite=False):
    """
    Processes a dataset by:
    - Copying validation & test images/labels without augmentation
    - Augmenting training images with copy-paste augmentation
    - Maintaining YOLO folder structure

    :param root_folder: Path to the original dataset (must contain 'train', 'valid', 'test' folders)
    :param dataset_destination_path: Path to save processed dataset
    :param num_copies: Number of augmented copies to generate for each training image
    :param overwrite: Whether to overwrite existing image
    """

    # Define dataset subfolders
    dataset_splits = ["train", "valid", "test"]

    # Create destination folders
    for split in dataset_splits:
        os.makedirs(os.path.join(dataset_destination_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_destination_path, split, "labels"), exist_ok=True)

    # copy data.yaml file
    shutil.copy(os.path.join(root_folder, "data.yaml"), os.path.join(dataset_destination_path, "data.yaml"))
    

    for split in ["valid", "test"]:
        img_folder = os.path.join(root_folder, split, "images")
        label_folder = os.path.join(root_folder, split, "labels")

        for img_file in os.listdir(img_folder):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(img_folder, img_file)
                label_path = os.path.join(label_folder, img_file.replace(".jpg", ".txt"))

                # Copy image
                img_save_path = os.path.join(dataset_destination_path, split, "images", img_file)
                cv2.imwrite(img_save_path, cv2.imread(img_path))

                # Copy label
                label_save_path = os.path.join(dataset_destination_path, split, "labels", img_file.replace(".jpg", ".txt"))
                with open(label_path, "r") as f_src, open(label_save_path, "w") as f_dst:
                    for line in f_src:
                        values = line.strip().split()
                        class_id = int(float(values[0]))  # Ensure class ID is an integer
                        bbox_values = [f"{float(v)}" for v in values[1:]] 
                        f_dst.write(f"{class_id} " + " ".join(bbox_values) + "\n")


    train_img_folder = os.path.join(root_folder, "train", "images")
    train_label_folder = os.path.join(root_folder, "train", "labels")

    # Extract ball objects once
    ball_objects = extract_ball_objects(train_img_folder, train_label_folder)
    print(f"Extracted {len(ball_objects)} ball objects for augmentation.")

    for img_file in os.listdir(train_img_folder):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(train_img_folder, img_file)
            label_path = os.path.join(train_label_folder, img_file.replace(".jpg", ".txt"))

            img = cv2.imread(img_path)

            # Load labels
            labels = []
            with open(label_path, "r") as f:
                labels = [list(map(float, line.strip().split())) for line in f.readlines()]

            # Save original image & labels
        
            img_save_path = os.path.join(dataset_destination_path, "train", "images", img_file)
            label_save_path = os.path.join(dataset_destination_path, "train", "labels", img_file.replace(".jpg", ".txt"))

            
            if not overwrite:
                    cv2.imwrite(img_save_path, img)
                    with open(label_path, "r") as f_src, open(label_save_path, "w") as f_dst:
                        for line in f_src:
                            values = line.strip().split()
                            class_id = int(float(values[0]))  # Ensure class ID is an integer
                            bbox_values = [f"{float(v)}" for v in values[1:]]
                            f_dst.write(f"{class_id} " + " ".join(bbox_values) + "\n")

            # Apply augmentation 
            for i in range(num_copies):
                
                img_aug, labels_aug = copy_paste_augmentation(img, labels, ball_objects)
                aug_img_file = f"aug_{i}_{img_file}"
                aug_label_file = aug_img_file.replace(".jpg", ".txt")

                aug_img_save_path = os.path.join(dataset_destination_path, "train", "images", aug_img_file)
                aug_label_save_path = os.path.join(dataset_destination_path, "train", "labels", aug_label_file)

                cv2.imwrite(aug_img_save_path, img_aug)
                with open(aug_label_save_path, "w") as f:
                    for label in labels_aug:
                        class_id = int(label[0])  # Ensure integer class ID
                        bbox_values = [f"{float(v)}" for v in label[1:]]
                        f.write(f"{class_id} " + " ".join(bbox_values) + "\n")


            else: 
                cv2.imwrite(img_save_path, img)
                with open(label_path, "r") as f_src, open(label_save_path, "w") as f_dst:
                        for line in f_src:
                            values = line.strip().split()
                            class_id = int(float(values[0]))  # Ensure class ID is an integer
                            bbox_values = [f"{float(v)}" for v in values[1:]] 
                            f_dst.write(f"{class_id} " + " ".join(bbox_values) + "\n")

    print("Dataset processing completed successfully!")
    


    
    


# class CustomYOLODataset(Dataset):
#     def __init__(self, image_folder, label_folder, transform=None):
#         self.image_folder = image_folder
#         self.label_folder = label_folder
#         self.image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
#         self.transform = transform
#         self.ball_objects = extract_ball_objects(image_folder, label_folder)  # Extract balls

#     def __getitem__(self, index):
#         # Load image
#         img_path = os.path.join(self.image_folder, self.image_files[index])
#         img = cv2.imread(img_path)
        
#         # Load label
#         label_path = os.path.join(self.label_folder, self.image_files[index].replace(".jpg", ".txt"))
#         labels = []
#         with open(label_path, "r") as f:
#             for line in f.readlines():
#                 labels.append(list(map(float, line.strip().split())))

#         # Apply Copy-Paste Augmentation
#         img, labels = copy_paste_augmentation(img, labels, self.ball_objects)

#         # Convert image to PyTorch Tensor
#         if self.transform:
#             img = self.transform(img)

#         return img, labels

#     def __len__(self):
#         return len(self.image_files)

# # Define image transformations (Normalization, Resize)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((640, 640)),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

# if __name__ == "__main__":
#     bbox_copy_paste('dataset/yolov8/v3', 'dataset/yolov8/v3_copy_paste_aug', 0.5, overwrite=True)
    

