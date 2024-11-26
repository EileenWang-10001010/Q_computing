import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
                                     UpSampling2D, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def iou_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + K.epsilon()) / (union + K.epsilon())

def precision(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    return tp / (tp + fp + K.epsilon())

def recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))
    return tp / (tp + fn + K.epsilon())

def bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def binary_segnet(IMG_CHANNELS, LearnRate):
    inputs = Input(shape=(None, None, IMG_CHANNELS))
    
    # Downsampling
    c1 = Conv2D(16, (3, 3), padding='same', kernel_initializer='glorot_uniform')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), padding='same', kernel_initializer='glorot_uniform')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_uniform')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_uniform')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_uniform')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Dropout(0.1)(c3)
    c3 = Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_uniform')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_uniform')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Dropout(0.1)(c4)
    c4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_uniform')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_uniform')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Dropout(0.1)(c5)
    c5 = Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_uniform')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    
    # Upsampling
    u4 = UpSampling2D((2, 2))(c5)
    u4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_uniform')(u4)
    u4 = BatchNormalization()(u4)
    u4 = Activation('relu')(u4)
    u4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_uniform')(u4)
    u4 = BatchNormalization()(u4)
    u4 = Activation('relu')(u4)
    
    u3 = UpSampling2D((2, 2))(u4)
    u3 = Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_uniform')(u3)
    u3 = BatchNormalization()(u3)
    u3 = Activation('relu')(u3)
    u3 = Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_uniform')(u3)
    u3 = BatchNormalization()(u3)
    u3 = Activation('relu')(u3)
    
    u2 = UpSampling2D((2, 2))(u3)
    u2 = Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_uniform')(u2)
    u2 = BatchNormalization()(u2)
    u2 = Activation('relu')(u2)
    u2 = Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_uniform')(u2)
    u2 = BatchNormalization()(u2)
    u2 = Activation('relu')(u2)
    
    u1 = UpSampling2D((2, 2))(u2)
    u1 = Conv2D(16, (3, 3), padding='same', kernel_initializer='glorot_uniform')(u1)
    u1 = BatchNormalization()(u1)
    u1 = Activation('relu')(u1)
    u1 = Conv2D(16, (3, 3), padding='same', kernel_initializer='glorot_uniform')(u1)
    u1 = BatchNormalization()(u1)
    u1 = Activation('relu')(u1)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u1)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=LearnRate), 
                  loss=bce_dice_loss, 
                  metrics=['binary_accuracy', iou_coefficient, dice_coef, precision, recall])
    model.summary()
    return model

IMG_CHANNELS = 3
LearnRate = 0.001
model = binary_segnet(IMG_CHANNELS, LearnRate)


X = np.load("images.npy").astype("float32") # (n, 64, 64, 3)
y = np.load("masks.npy").astype("float32") # (n, 64, 64, 1)


indices = np.arange(len(X))
X_train, X_val, y_train, y_val, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.2, random_state=7)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=64
)

results = model.evaluate(X_val, y_val, verbose=1)

loss = results[0]
accuracy = results[1]
iou = results[2]
dice = results[3]
precisions = results[4]
recalls = results[5]

print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"IoU: {iou:.4f}")
print(f"Dice Coefficient: {dice:.4f}")
print(f"Precision: {precisions:.4f}")
print(f"Recall: {recalls:.4f}")

print(history.history.keys())

# Name = "C_SegNet_E40_B64"
# os.makedirs(f"result_img/{Name}", exist_ok=True)
# np.save(f"result_img/{Name}/{Name}.npy",history.history)


def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coef'], label='Training Dice Coefficient')
    plt.plot(history.history['val_dice_coef'], label='Validation Dice Coefficient')
    plt.title('Dice Coefficient Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    plt.tight_layout()
    plt.show()
    # plt.savefig(f"result_img/{Name}/{Name}.png")
    # plt.close()

plot_training_history(history)


def plot_input_ground_truth_prediction(X_test, y_test, y_pred_bin, index=0):

    # X_test (n, 64, 64, 3)
    # y_test (n, 64, 64, 1)
    # y_pred_bin (n, 64, 64, 1)

    input_image = X_test[index]
    ground_truth = y_test[index].squeeze()  
    prediction = y_pred_bin[index].squeeze()  

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow((input_image).astype(np.uint8))
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    # plt.savefig(f"result_img/{Name}/{index}.png")
    # plt.close()

y_pred = model.predict(X_val)
y_pred_bin = (y_pred > 0.5).astype(np.uint8)


lst = [0,1,7, 17, 197, 211, 296, 298]
# [ 547  984 1158  747   51  259  518 1430]

seq = np.array(test_indices)[lst]

for i in lst:
    plot_input_ground_truth_prediction(X_val, y_val, y_pred_bin, i)