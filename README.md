# TrackTraffic

## Overview

The TrackTraffic project aims to use Convolutional Neural Networks (CNN) to detect cars and motorcycles from a video capture. The project consists of two main Python scripts, `cnn.py` and `ContourDetect.py`. The former involves training a CNN model using Keras to classify images into two classes: cars and motorcycles. The latter utilizes OpenCV to perform background subtraction, detect contours, and count the number of vehicles in a video stream.

## Cnn.py

### Dependencies
- Keras
- TensorFlow
- Matplotlib
- NumPy
- OpenCV

### Code Structure
1. **Data Preprocessing:**
   - Image data generators are used to preprocess the training and validation datasets, rescaling pixel values to a range of [0, 1].
   - Training and validation datasets are created using the `flow_from_directory` method from Keras.

2. **CNN Model:**
   - A sequential model is constructed using Keras layers, including convolutional layers, max-pooling layers, flattening layer, and dense layers.
   - The model is compiled with the sparse categorical cross-entropy loss function, Adam optimizer, and accuracy metric.

3. **Model Training:**
   - The CNN model is trained on the training dataset with 30 epochs and 10 steps per epoch.
   - Validation data is used to evaluate the model's performance during training.

4. **Image Prediction:**
   - Images from the testing dataset are loaded, and predictions are made using the trained CNN model.
   - Predicted values are used to display whether an image contains a car or a motorcycle.

### Usage
- Ensure the required dependencies are installed.
- Organize the dataset into appropriate directories (e.g., `data/training`, `data/validation`, `data/testing`).
- Run the script to train the CNN model and make predictions on test images.

## ContourDetect.py

### Dependencies
- OpenCV

### Code Structure
1. **Video Capture:**
   - OpenCV is used to capture video from a specified file (`video2.mp4`).
   - Background subtraction is applied to isolate moving objects.

2. **Region of Interest (ROI) Selection:**
   - A specific region of the frame is defined as the area of interest to reduce processing load.

3. **Background Subtraction:**
   - The k-Nearest Neighbors background subtractor is employed to distinguish foreground objects.

4. **Thresholding and Morphological Operations:**
   - Thresholding is applied to the mask, and morphological operations (dilation) help refine object boundaries.

5. **Contour Detection:**
   - Contours of objects are detected using OpenCV's `findContours` function.

6. **Vehicle Counting:**
   - Valid vehicles are identified based on size criteria, and their count is displayed on the frame.

### Usage
- Ensure OpenCV is installed.
- Provide the correct video file path in the script.
- Run the script to perform vehicle detection in the specified video.

## Notes
- These scripts serve as components for vehicle detection in a larger system.
- Fine-tuning parameters may be required based on specific use cases and datasets.
- Proper organization of the dataset directory is required for the CNN model.
