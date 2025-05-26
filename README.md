# Plant-leaf-disease-detection
Great! Here's a detailed **README.md** content for your Plant Leaf Disease Detection GitHub repo, including overview, installation, usage, model descriptions, and example code snippets for training and inference.

---

# Plant Leaf Disease Detection

## Overview

This project aims to detect and classify diseases in plant leaves using advanced deep learning and machine learning models. Accurate identification helps farmers and researchers take timely actions to protect crops and improve yield.

## Features

* Multiple models for disease detection, including CNNs and transfer learning architectures.
* Preprocessing and augmentation pipeline.
* Training and evaluation scripts.
* Support for classical ML models using extracted features.
* Ensemble learning for improved accuracy.
* Visualization of predictions and performance metrics.

## Models Included

| Model                            | Description                                       |
| -------------------------------- | ------------------------------------------------- |
| CNN (From Scratch)               | Custom convolutional neural network               |
| ResNet50                         | Fine-tuned ResNet50 pretrained on ImageNet        |
| VGG16 / VGG19                    | Deep networks fine-tuned for leaf disease dataset |
| InceptionV3                      | Efficient deep learning architecture              |
| MobileNet                        | Lightweight CNN for mobile and edge devices       |
| DenseNet                         | Dense convolutional network for feature reuse     |
| Classical ML (SVM, RF, KNN, GBM) | Trained on extracted image features               |
| Ensemble                         | Combination of multiple model predictions         |

## Dataset

The Plant Village dataset is used, which contains labeled images of healthy and diseased leaves from various plants.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/plant-leaf-disease-detection.git
cd plant-leaf-disease-detection

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Data Preparation

Place your dataset in the `data/` directory, organized as:

```
data/
  train/
    healthy/
    disease1/
    disease2/
    ...
  test/
    healthy/
    disease1/
    disease2/
    ...
```

### 2. Training a CNN Model

Example training script using Keras:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Data generators with augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

val_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_directory(
    'data/test',
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator,
          epochs=15,
          validation_data=validation_generator)

model.save('cnn_leaf_disease_model.h5')
```

### 3. Using Transfer Learning (ResNet50 example)

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Use similar data generators but with target size 224x224
train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory('data/train', target_size=(224,224), batch_size=32, class_mode='categorical')
validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory('data/test', target_size=(224,224), batch_size=32, class_mode='categorical')

model.fit(train_generator, epochs=10, validation_data=validation_generator)
```

### 4. Predicting on New Images

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('cnn_leaf_disease_model.h5')

img_path = 'sample_leaf.jpg'
img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
class_idx = np.argmax(pred, axis=1)[0]
print(f"Predicted class index: {class_idx}")
```
