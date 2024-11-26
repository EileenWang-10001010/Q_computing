import os
import cv2
import numpy as np
from PIL import Image
import torch
import pandas as pd
import time
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

n_epochs = 20
np.random.seed(0)   

# load original images
Root = "/home/tester/Ai_Lin"
name_list = ["eos_0.3to0.4"]

img_stack = []
label_stack = []

for idx, name in enumerate(name_list):
    filename = np.loadtxt(f"{Root}/data/cell/filenameList/{name}.txt", dtype=str)
    # print(len(filename))
    folder = "Lymphocyte" if idx else "Eosinophil"
    for img_name in filename:
        img = Image.open(f"{Root}/{folder}/{img_name}")
        img_stack.append(np.array(img))
    label_stack.extend([idx]*len(filename))

img_stack = np.array(img_stack)
# scaling
img_stack = img_stack/255.0
label_stack = np.array(label_stack)

print(img_stack.shape,label_stack.shape)


# load quantum feature npy stacks
q_PATH = "result/q_data"
q_img_stack = np.load(f"{q_PATH}/{name_list[0]}.npy")

# train test split
train_images, test_images, train_labels, test_labels = train_test_split(
    img_stack, label_stack, test_size=0.25, random_state=7)

q_train_images, q_test_images, q_train_labels, q_test_labels = train_test_split(
    q_img_stack, label_stack, test_size=0.25, random_state=7)

# model
def Model(input_shape):
    model = keras.models.Sequential([

        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

q_model = Model((32, 32, 12))

q_history = q_model.fit(
    q_train_images,
    train_labels,
    validation_data=(q_test_images, test_labels),
    batch_size=8,
    epochs=n_epochs,
    verbose=2,
)

c_model = Model((64, 64, 3))

c_history = c_model.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    batch_size=8,
    epochs=n_epochs,
    verbose=2,
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

y_pred = q_model.predict(q_test_images)
y_pred = np.argmax(y_pred, axis=1)

conf_mat = confusion_matrix(q_test_labels, y_pred)
print("q data model performance", conf_mat, sep="\n")

y_pred = c_model.predict(test_images)
y_pred = np.argmax(y_pred, axis=1)

conf_mat = confusion_matrix(test_labels, y_pred)
print("c data model performance", conf_mat, sep="\n")


ax1.plot(q_history.history["val_accuracy"], "-ob", label="With quantum layer")
ax1.plot(c_history.history["val_accuracy"], "-og", label="Without quantum layer")
ax1.set_ylabel("Accuracy")
ax1.set_ylim([0, 1])
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(q_history.history["val_loss"], "-ob", label="With quantum layer")
ax2.plot(c_history.history["val_loss"], "-og", label="Without quantum layer")
ax2.set_ylabel("Loss")
ax2.set_ylim(top=2.5)
ax2.set_xlabel("Epoch")
ax2.legend()

plt.savefig("val_loss.png")