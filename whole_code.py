#Basic Setup & Imports
import os
from pathlib import Path
import random
import hashlib
from collections import defaultdict
from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB4
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

# 2: Load Dataset & Scan Directory Structure
from pathlib import Path

dataset_path = Path('/content/drive/MyDrive/data')

#Folder Structure Inspection
for dirpath, dirnames, filenames in os.walk(dataset_path):
    print(f"Found directory: {dirpath} with {len(filenames)} images")

print("\nTop-level folders (classes):", os.listdir(dataset_path))

#Show 1 sample image per class
sample_images = []
image_extensions = ['.jpg', '.jpeg', '.png']

categories = os.listdir(dataset_path)
for category in categories:
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in image_extensions):
                sample_images.append(file_path)
                break  # only 1 sample per category

plt.figure(figsize=(12, 5))
for i, img_path in enumerate(sample_images):
    img = Image.open(img_path)
    plt.subplot(1, len(sample_images), i + 1)
    plt.imshow(img)
    plt.title(os.path.basename(os.path.dirname(img_path)))
    plt.axis('off')
plt.show()

#Data Preprocessing & Augmentation
BATCH_SIZE = 32
IMG_SIZE = (224, 224)  # standard for CNNs
VAL_SPLIT = 0.2
SEED = 123

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

#Normalize (Rescale [0, 1])
normalization_layer = layers.Rescaling(1./255)

#Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),  # ±10%
    layers.RandomZoom(0.1)
])

# Apply normalization & augmentation to training set
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y),
                        num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y),
                    num_parallel_calls=AUTOTUNE)

# Prefetch for performance
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Verify One Batch
for images, labels in train_ds.take(1):
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")



# 1) Check Rescaling [0, 1]
for images, labels in train_ds.take(1):
    min_val = images.numpy().min()
    max_val = images.numpy().max()
    print(f"Pixel Value Range: min={min_val:.4f}, max={max_val:.4f}")
    if 0.0 <= min_val and max_val <= 1.0:
        print("Images are correctly rescaled to [0, 1]")
    else:
        print("Rescaling issue: values outside [0, 1]")
    break

# 2) Check Augmentation Effects
# We're taking one image and show multiple augmented versions
def visualize_augmentation(dataset, num_examples=5):
    for images, labels in dataset.take(1):
        image = images[0].numpy()  # take first image in batch
        plt.figure(figsize=(15, 3))
        for i in range(num_examples):
            # Apply augmentation again (simulate training-time augmentation)
            augmented_img = data_augmentation(tf.expand_dims(image, 0), training=True)
            plt.subplot(1, num_examples, i + 1)
            plt.imshow(augmented_img[0].numpy())
            plt.axis('off')
        plt.suptitle("Augmented Examples of the Same Image")
        plt.show()
        break

visualize_augmentation(train_ds)

# Data Vizualization, Storytelling & Experimenting with charts 
def load_image_info(base_folder, set_type):
    data = []
    set_folder = os.path.join(base_folder, set_type) # Construct path to train/val folder
    if os.path.isdir(set_folder):
        for class_name in os.listdir(set_folder): # Iterate through class folders inside train/val
            class_path = os.path.join(set_folder, class_name)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        img = Image.open(img_path)
                        width, height = img.size
                        file_size = os.path.getsize(img_path) / 1024  # KB
                        data.append([class_name, width, height, width/height, file_size, set_type])
                    except:
                        pass
    return data

# Define the base dataset path
dataset_path = "/content/drive/MyDrive/data"

train_data = load_image_info(dataset_path, "train")
val_data = load_image_info(dataset_path, "val") # Use "val" as the folder name

df = pd.DataFrame(train_data + val_data, columns=["Class", "Width", "Height", "Aspect_Ratio", "File_Size_KB", "Set"])
print("Dataset Info:", df.shape)

#Chart 1 : Class Distribution Percentage
# Chart - 1 visualization code
plt.figure(figsize=(8,8))
df['Class'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Class Distribution Percentage")
plt.ylabel("")
plt.show()

#Chart - 2 : File Size Distribution by Class
# Chart - 2 visualization code
import seaborn as sns
plt.figure(figsize=(12,6))
sns.boxplot(x='Class', y='File_Size_KB', data=df)
plt.xticks(rotation=45)
plt.title("File Size Distribution by Class")
plt.show()

#Chart - 3 : Train vs Validation Image Count per Class
# Chart - 3 visualization code
train_counts = df[df['Set']=="train"]['Class'].value_counts()
val_counts = df[df['Set']=="validation"]['Class'].value_counts()
counts_df = pd.DataFrame({'Train': train_counts, 'Validation': val_counts})

counts_df.plot(kind='bar', figsize=(12,6), stacked=True)
plt.title("Train vs Validation Image Count per Class")
plt.ylabel("Image Count")
plt.xticks(rotation=45)
plt.show()

#Chart - 4 : Number of Images per Class
# Chart - 4 visualization code
plt.figure(figsize=(10,5))
df['Class'].value_counts().plot(kind='bar')
plt.title("Number of Images per Class")
plt.xticks(rotation=45)
plt.show()

#ML Model Implementation
# Paths
train_dir = "/content/drive/MyDrive/data/train"
val_dir = "/content/drive/MyDrive/data/val"

# Data Generators
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1./255)


train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)


# Function to train and evaluate models
def train_and_evaluate(base_model, model_name):
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(train_data.num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_data, validation_data=val_data, epochs=5)

    val_preds = model.predict(val_data)
    y_pred = np.argmax(val_preds, axis=1)
    # Get true labels from the generator
    val_data.reset() # Reset the generator to ensure consistent order
    y_true = val_data.classes[val_data.index_array]

    print(f"Classification Report for {model_name}:")
    # Ensure target_names match the order of classes in y_true
    print(classification_report(y_true, y_pred, target_names=list(val_data.class_indices.keys())))

    return model, history

# Models to test
models_to_test = {
    "VGG16": VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,)),
    "ResNet50": ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,)),
    "MobileNet": MobileNet(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,)),
    "InceptionV3": InceptionV3(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,)),
    "EfficientNetB4": EfficientNetB4(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
}

# Train all and store results
results = {}
for name, base in models_to_test.items():
    print(f"\n===== Training {name} =====")
    trained_model, history = train_and_evaluate(base, name)
    results[name] = {
        "model": trained_model,
        "history": history
    }

# Visualize accuracy curves
plt.figure(figsize=(12, 6))
for name, res in results.items():
    plt.plot(res["history"].history['val_accuracy'], label=name)
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

best_model = results["EfficientNetB4"]["model"]

# Unfreeze top layers for fine-tuning
best_model.layers[0].trainable = True

best_model.compile(optimizer=Adam(learning_rate=1e-5),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

history_finetune = best_model.fit(train_data, validation_data=val_data, epochs=5)

# Save best model
best_model.save("/content/drive/MyDrive/best_fish_model.h5")

metrics_summary = []
for name, res in results.items():
    acc = max(res["history"].history['val_accuracy'])
    metrics_summary.append({"Model": name, "Best Val Accuracy": acc})

metrics_df = pd.DataFrame(metrics_summary)
print(metrics_df.sort_values(by="Best Val Accuracy", ascending=False))

#1. Save the best performing ml model in a pickle file or joblib file format for deployment process.
# Save the File

# === 1. Save the best performing model (EfficientNetB4) ===
best_model_path = "best_fish_model.h5"
best_model.save(best_model_path)
print(f"Model saved to {best_model_path}")

# Also save a pickle file for any preprocessing objects if needed
# Example: Label encoder
# joblib.dump(label_encoder, "label_encoder.pkl")

#2. Again Load the saved model file and try to predict unseen data for a sanity check.



# Define best_model_path here
best_model_path = "/content/drive/MyDrive/best_fish_model.h5" # Assuming this is where you saved your model

# Load the trained model
loaded_model = load_model(best_model_path)
print("Model loaded successfully!")

# === 3. Sanity check prediction on unseen data ===
# Example unseen image path
test_img_path = "/content/drive/MyDrive/data/test/animal_fish/0JESIL2U7PFG.jpg" # Corrected path

# Preprocess image
img = image.load_img(test_img_path, target_size=(224, 224))  # Use the same size as trained
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
predictions = loaded_model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Map back to label
# class_labels = list(label_encoder.classes_) # Need label encoder or class_indices from generator
# print("Predicted Class:", class_labels[predicted_class[0]])
print("Predicted Class Index:", predicted_class[0])
