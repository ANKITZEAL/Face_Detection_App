import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

def load_and_display_images(dataset_path, num_persons=5, num_images=15):
    # Ensure the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} not found.")
    
    # List all subdirectories (each person's name)
    persons = os.listdir(dataset_path)
    if len(persons) == 0:
        raise ValueError("No persons found in the dataset path.")

    # Function to display properties of sample images
    def display_sample_images(person_name):
        person_images = os.listdir(os.path.join(dataset_path, person_name))
        sample_images = person_images[:5]  # Load first 5 images
        image_properties = []
        for img_name in sample_images:
            img_path = os.path.join(dataset_path, person_name, img_name)
            img = Image.open(img_path)
            image_properties.append((img_name, img.size, img.format))
        return image_properties
    
    # Select and display images from random persons
    def display_random_person_images(persons, num_persons):
        random_persons = random.sample(persons, num_persons)
        selected_images = []
        for person in random_persons:
            person_image_dir = os.path.join(dataset_path, person)
            person_image_files = os.listdir(person_image_dir)
            selected_image_file = random.choice(person_image_files)
            selected_images.append(os.path.join(person_image_dir, selected_image_file))
        
        fig, axs = plt.subplots(1, num_persons, figsize=(15, 3))
        for i, img_path in enumerate(selected_images):
            img = mpimg.imread(img_path)
            axs[i].imshow(img)
            axs[i].set_title(os.path.basename(os.path.dirname(img_path)))  # Title with the person's name
            axs[i].axis('off')
        plt.show()

    # Display a grid of images to show diversity
    def display_grid_images(persons, num_images):
        random_persons_sample = np.random.choice(persons, num_images, replace=False)
        selected_sample_images = []
        for person in random_persons_sample:
            person_image_dir = os.path.join(dataset_path, person)
            person_image_files = os.listdir(person_image_dir)
            selected_image_file = random.choice(person_image_files)
            selected_sample_images.append(os.path.join(person_image_dir, selected_image_file))
        
        fig, axs = plt.subplots(3, 5, figsize=(15, 9))
        for i, img_path in enumerate(selected_sample_images):
            img = mpimg.imread(img_path)
            axs[i // 5, i % 5].imshow(img)
            axs[i // 5, i % 5].set_title(os.path.basename(os.path.dirname(img_path)))
            axs[i // 5, i % 5].axis('off')
        plt.tight_layout()
        plt.show()
    
    # Display properties of sample images from a randomly selected person
    selected_person = random.choice(persons)
    image_properties = display_sample_images(selected_person)
    print(f"Sample images properties from person '{selected_person}': {image_properties}")

    # Display images from random persons
    display_random_person_images(persons, num_persons)
    
    # Display a grid of images
    display_grid_images(persons, num_images)

# Usage
extract_to_path = 'C:/Users/ANKIT/Downloads/archive/lfw-deepfunneled/lfw-deepfunneled'
load_and_display_images(extract_to_path)
