
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")


# Quantum setup
n_qubits = 4
n_quantum_outputs = 8  
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf")  
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.BasicEntanglerLayers(qml.numpy.array(weights), wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (3, n_qubits)}


class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(QuantumLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.quantum_weights = self.add_weight(
            name="quantum_weights",
            shape=weight_shapes["weights"],
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        def quantum_fn(x):
            result = quantum_circuit(x, qml.numpy.array(self.quantum_weights))
            return np.array(result, dtype=np.float32)

        quantum_results = tf.map_fn(
            lambda x: tf.numpy_function(quantum_fn, [x], tf.float32),
            inputs
        )
        quantum_results.set_shape([None, n_qubits]) 
        return quantum_results

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-7) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-7)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

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

def quantum_unet(IMG_CHANNELS, LearnRate):
    inputs = Input((64, 64, IMG_CHANNELS))

    c1 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
    c3 = Dropout(0.1)(c3)
    c3 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
    c4 = Dropout(0.1)(c4)
    c4 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Quantum layer
    quantum_input = Flatten()(p4) 
    quantum_dense = Dense(n_qubits, activation="relu")(quantum_input)  
    quantum_output = QuantumLayer()(quantum_dense)                     
    quantum_output = Dense(128, activation="relu")(quantum_output)      
    quantum_output = Reshape((4, 4, 8))(quantum_output)                  

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(quantum_output)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
    c6 = Dropout(0.1)(c6)
    c6 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=LearnRate), loss=bce_dice_loss, metrics=['binary_accuracy', iou_coefficient, dice_coef, precision, recall])
    return model

IMG_CHANNELS = 3
LearnRate = 0.001
model = quantum_unet(IMG_CHANNELS, LearnRate)

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

# Name = "Q_DeepUNet_E40_B64"
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