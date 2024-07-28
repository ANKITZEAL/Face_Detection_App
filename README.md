# Face Detection and Recognition Project 
![image](https://github.com/user-attachments/assets/44d3f6fe-68d7-4dc8-806f-8ba6e3d6b1a0)


## Project Overview

This project develops a face detection and recognition system using machine learning techniques. It employs OpenCV for face detection and TensorFlow-based models for face recognition. The primary dataset used is the Labeled Faces in the Wild (LFW) dataset, which provides a diverse collection of unconstrained face photographs.

## Dataset

### Context

The Labeled Faces in the Wild (LFW) dataset comprises 13,233 images of 5,749 individuals, designed to facilitate the study of face recognition under various conditions. Images have been pre-processed with the Viola-Jones face detector and aligned using a deep funneling technique to enhance recognition performance.

### Content

- **Image Files**: Stored in the format `lfw/name/name_xxxx.jpg`, where `xxxx` is a zero-padded image number.
- **Image Dimensions**: Each image is 250x250 pixels.
- **Metadata**: Includes various files to assist in forming training and testing sets.

## Project Structure

![image](https://github.com/user-attachments/assets/db62e80e-a063-4155-bfd1-fd06177f117c)


- **data/**: Contains the precomputed embeddings and labels.
- **models/**: Contains the face detection model files.
- **utils/**: Includes utility scripts for face detection and recognition.
- **app.py**: Streamlit application for face detection and recognition.
- **scale.py**: Script for training and evaluating the face recognition model.
- **dataset.py**: Script to load and preprocess the LFW dataset.
- **requirements.txt**: Lists the project dependencies.

## Installation

1. **Create a Virtual Environment:**

   ```bash
   python -m venv env
   ```

2. **Activate the Virtual Environment:**
```bash
.\env\Scripts\activate
```

3. **Install the Dependencies:**
```bash
pip install -r requirements.txt
```
## Additional Setup Instructions

### Dataset Preparation

1. **Download the LFW Dataset:**

   - Obtain the LFW dataset from [here](http://vis-www.cs.umass.edu/lfw/).
   - Place the dataset in the specified directory of your project.

2. **Prepare Embeddings and Labels:**

   - Ensure that the precomputed embeddings and labels are saved in the `data/` directory. This involves running the preprocessing scripts to generate these files if they are not already present.

### Model Files

1. **Download Face Detection Model Files:**

   - Download the face detection model files, specifically `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`.

2. **Place the Model Files:**

   - Place these files in the `models/` directory of your project to enable face detection functionality.


## Running the Streamlit App

To run the Streamlit application and interact with the face detection and recognition system, use the following command:

```bash
streamlit run app.py
```
## Summary

This project integrates face detection and recognition models using OpenCV and TensorFlow, with a focus on achieving high accuracy and robust performance. It covers the complete workflow from data loading and preprocessing to model training, evaluation, and deployment via a user-friendly Streamlit application.

### Features

- **Face Detection**: Utilizes OpenCV’s deep learning-based face detector to accurately locate faces within various image conditions.
- **Face Recognition**: Employs a machine learning model to recognize and identify individuals based on facial features. The recognition process includes extracting facial embeddings and comparing them against known faces.
- **Confidence Scores**: The Streamlit application not only detects and recognizes faces but also provides confidence scores for each recognition result. These scores indicate the model's certainty about its predictions, helping users assess the reliability of the recognition outcomes.

Initial performance challenges have been identified, and there are opportunities for further optimization and enhancement. The project provides a comprehensive solution for face detection and recognition, leveraging state-of-the-art techniques and tools to handle real-world, unconstrained images effectively.



![image](https://github.com/user-attachments/assets/7e35855e-5adc-4a04-8403-e114b1b3af31)

