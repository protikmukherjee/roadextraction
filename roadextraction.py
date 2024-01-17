#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:43:53 2024

@author: protikmukherjee
"""

# In[1]:

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation, Add, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from PIL import Image



# In[2]:


def conv_block(input_tensor, num_filters):
    """Convolutional Block consisting of two convolutional layers."""
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


# In[3]:


class ResizeAndCropLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ResizeAndCropLayer, self).__init__(**kwargs)

    def call(self, inputs):
        input_tensor, gating_signal = inputs
        # Resize gating_signal to match input_tensor shape
        resized = tf.image.resize(gating_signal, (tf.shape(input_tensor)[1], tf.shape(input_tensor)[2]))
        
        # cropping
        cropped = resized[:, :tf.shape(input_tensor)[1], :tf.shape(input_tensor)[2], :]
        return cropped

def attention_block(input_tensor, gating_signal, num_filters):
    """Attention block for Model"""
    x = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
    x = BatchNormalization()(x)

    # Upsample and crop gating_signal to match input_tensor size using custom layer
    g = Conv2D(num_filters, (1, 1), padding='same')(gating_signal)
    g = BatchNormalization()(g)
    g = ResizeAndCropLayer()([input_tensor, g])

    x = Add()([x, g])
    x = Activation('relu')(x)

    x = Conv2D(1, (1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)

    return Multiply()([input_tensor, x])


# In[4]:


def REmodel(input_shape=(256, 256, 3)):
    """REmodel."""
    inputs = Input(input_shape)

    # Encoder
    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = conv_block(p4, 1024)

    # Decoder
    a4 = attention_block(c4, c5, 512)
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, a4])

    c6 = conv_block(u6, 512)

    a3 = attention_block(c3, c6, 256)
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, a3])

    c7 = conv_block(u7, 256)

    a2 = attention_block(c2, c7, 128)
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, a2])

    c8 = conv_block(u8, 128)

    a1 = attention_block(c1, c8, 64)
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, a1])

    c9 = conv_block(u9, 64)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


remodel = REmodel()


# In[5]:


def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def IoU(y_true, y_pred, smooth=1):
    threshold = 0.5
    y_pred = K.cast(y_pred > threshold, K.floatx())
    
    if K.ndim(y_true) == 3:
        y_true = K.expand_dims(y_true, -1)
    if K.ndim(y_pred) == 3:
        y_pred = K.expand_dims(y_pred, -1)

    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1,2,3])
    iou = (intersection + smooth) / (sum_ - intersection + smooth)
    return iou



remodel.compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy', IoU, Recall(), Precision()])




#%%
class TiffTifDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images_dir, labels_dir, batch_size, img_size):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.tiff')]
        self.label_extensions = ['.tif', '.tiff']

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = np.array([np.array(Image.open(os.path.join(self.images_dir, file_name))) for file_name in batch_x])

        # Adjusting for label files with '.tif' extension
        labels = []
        for file_name in batch_x:
            label_file_name = os.path.splitext(file_name)[0]  # Removing extension
            for ext in self.label_extensions:
                label_path = os.path.join(self.labels_dir, label_file_name + ext)
                if os.path.exists(label_path):
                    label_image = np.array(Image.open(label_path))
                    label_image[label_image > 0] = 1  # Convert to binary format
                    labels.append(label_image)
                    break

        labels = np.array(labels)

        # Resize images and labels if img_size is specified
        if self.img_size is not None:
            images = np.array([np.array(Image.fromarray(img).resize(self.img_size)) for img in images])
            labels = np.array([np.array(Image.fromarray(lbl).resize(self.img_size)) for lbl in labels])

        # Normalize images
        images = images.astype('float32') / 255.0

        # Ensuring labels are in 'float32' for compatibility
        labels = labels.astype('float32')

        return images, labels


# In[7]:


train_images_dir = '/Users/protikmukherjee/Work/train'
train_labels_dir = '/Users/protikmukherjee/Work/train_labels'
val_images_dir = '/Users/protikmukherjee/Work/val'
val_labels_dir = '/Users/protikmukherjee/Work/val_labels'
test_images_dir = '/Users/protikmukherjee/Work/test'
test_labels_dir = '/Users/protikmukherjee/Work/test_labels'

batch_size = 32
img_size = (256, 256) 

train_generator = TiffTifDataGenerator(train_images_dir, train_labels_dir, batch_size, img_size)
val_generator = TiffTifDataGenerator(val_images_dir, val_labels_dir, batch_size, img_size)
test_generator = TiffTifDataGenerator(test_images_dir, test_labels_dir, batch_size, img_size)

# train_generator = TiffTifDataGenerator(train_images_dir, train_labels_dir, batch_size, img_size, augment=True)
# val_generator = TiffTifDataGenerator(val_images_dir, val_labels_dir, batch_size, img_size, augment=False)
# test_generator = TiffTifDataGenerator(test_images_dir, test_labels_dir, batch_size, img_size, augment=False)

# In[ ]:


with tf.device("/device:GPU:0"):
    # Train
        history = remodel.fit(
        train_generator,
        validation_data=val_generator,
        epochs=130
    )
# Evaluate 
test_results = remodel.evaluate(test_generator, return_dict=True)
print("Test Metrics:", test_results)
    
   


# %%In[ ]:

from matplotlib import pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


#%% In[ ]:
# Plot for IoU
plt.figure(figsize=(12, 6))
plt.plot(history.history['IoU'])
plt.plot(history.history['val_IoU'])
plt.title('Model IoU')
plt.ylabel('IoU')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# Plot for Precision and Recall
plt.figure(figsize=(12, 6))
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('Precision and Recall')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(['Train Precision', 'Validation Precision', 'Train Recall', 'Validation Recall'], loc='upper left')
plt.show()

#%%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming you have your predictions and ground truth labels
# y_pred: model predictions, y_true: actual labels
y_pred = remodel.predict(test_generator)
y_true = np.concatenate([labels for _, labels in test_generator], axis=0)

# Flatten the arrays
y_pred = y_pred.flatten()
y_true = y_true.flatten()

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#%%
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate binary predictions
threshold = 0.5
y_pred_bin = (y_pred > threshold).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_bin)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def save_combined_comparisons(images_dir, labels_dir, model, img_size, output_file, num_samples=5):
    """
    Saves a single image file with multiple comparisons.

    :param images_dir: Directory containing the images.
    :param labels_dir: Directory containing the ground truth labels.
    :param model: Trained segmentation model.
    :param img_size: Size to which images should be resized.
    :param output_file: File path where the combined image will be saved.
    :param num_samples: Number of samples to display.
    """
    image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.tiff')]
    np.random.shuffle(image_filenames)

    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    for i, filename in enumerate(image_filenames[:num_samples]):
        img_path = os.path.join(images_dir, filename)
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0

        pred = model.predict(img)
        pred = pred.squeeze()

        label_filename = os.path.splitext(filename)[0] + '.tif'  # Adjust if necessary
        label_path = os.path.join(labels_dir, label_filename)
        ground_truth = load_img(label_path, target_size=img_size, color_mode='grayscale')
        ground_truth = img_to_array(ground_truth).squeeze() / 255.0

        axs[i, 0].imshow(img.squeeze())
        axs[i, 0].set_title('Original Image')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(ground_truth, cmap='gray')
        axs[i, 1].set_title('Ground Truth')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(pred, cmap='gray')
        axs[i, 2].set_title('Prediction')
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig)

# Example usage
save_combined_comparisons(train_images_dir, train_labels_dir, remodel, img_size, '/Users/protikmukherjee/Downloads/combined_comparison1.png')

#%%
# Evaluate the model on the test set
test_results = remodel.evaluate(test_generator, return_dict=True)

# Display test results
print("Test Metrics:", test_results)

#%%
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import random

def load_and_preprocess_image(image_path, label_path, img_size):
    # Load image
    img = Image.open(image_path)
    img = img.resize(img_size)
    img = np.array(img) / 255.0  # Normalize

    # Load label
    label = Image.open(label_path)
    label = label.resize(img_size)
    label = np.array(label) / 255.0

    # Assuming labels are binary (0 or 1)
    label[label > 0] = 1

    return img, label

def display_samples(images_dir, labels_dir, img_size, num_samples=3):
    image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.tiff')]
    random.shuffle(image_filenames)  # Shuffle the list to pick random samples
    selected_filenames = image_filenames[:num_samples]  # Pick the first three after shuffling

    fig, axs = plt.subplots(num_samples, 3, figsize=(20, num_samples * 6))

    for i, file_name in enumerate(selected_filenames):
        img_path = os.path.join(images_dir, file_name)
        label_path = os.path.join(labels_dir, file_name.replace('.tiff', '.tif'))  

        img, label = load_and_preprocess_image(img_path, label_path, img_size)

        axs[i, 0].imshow(img)
        axs[i, 0].set_title('Original Image')
        axs[i, 0].axis('off')
        axs[i, 0].set_aspect('auto')

        axs[i, 1].imshow(img)
        axs[i, 1].set_title('Preprocessed Image')
        axs[i, 1].axis('off')
        axs[i, 1].set_aspect('auto')

        axs[i, 2].imshow(label.squeeze(), cmap='gray')
        axs[i, 2].set_title('Label')
        axs[i, 2].axis('off')
        axs[i, 2].set_aspect('auto')

    plt.tight_layout(pad=3.0)
    plt.show()

# Directories of images and labels
images_dir = '/Users/protikmukherjee/Work/train' 
labels_dir = '/Users/protikmukherjee/Work/train_labels'  

# Size of the images for preprocessing
img_size = (256, 256)  

# Display the samples
display_samples(images_dir, labels_dir, img_size)

#%%
from tensorflow.keras.utils import plot_model

model=REmodel()

plot_model(model, to_file='unet_model.png', show_shapes=True, show_layer_names=True)



