#  EfficientNetB4 Image Classification with Grad-CAM

This repository contains an **image classification pipeline** using **EfficientNetB4** in TensorFlow/Keras, with **Grad-CAM visualization** to highlight the image regions most important for the model's prediction.  

It can be used for any classification task — in this example, we demonstrate with an **animal classification dataset**.

---

##  Features
- Load a pre-trained EfficientNetB4 model
- Predict on single or multiple images
- Visualize Grad-CAM heatmaps to understand model decisions
- Fully customizable for your own dataset

---

##  Project Structure

├── model/ # Directory containing the trained model (.h5)
├── data/ # Dataset directory
│ ├── train/
│ ├── val/
│ └── test/
├── gradcam.py # Grad-CAM visualization script
├── predict.py # Prediction script
├── class_labels.pkl # Pickled list of class names
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── examples/ # Example images and outputs