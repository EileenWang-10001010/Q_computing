import os
import cv2
import torch
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import time
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

start = time.time()

n_layers = 1    # Number of random layers

SAVE_PATH = "result/q_data"  
PREPROCESS = True           
np.random.seed(0)           

PATH = "/home/tester/Ai_Lin/"

images = []

# file name list
file_list = np.loadtxt(PATH+"data/cell/filenameList/eos_0.3to0.4.txt", dtype=str)

# stack images for quantum feature generation
for  file in file_list:
        tmp = cv2.imread(f"{PATH}Eosinophil/{file}")
        images.append(tmp)


images = np.array(images)
images = images/255.0
print(len(images))

train_images = torch.tensor(images, dtype=torch.float32).to('cuda:0' if torch.cuda.is_available() else 'cpu')

n_train = len(train_images)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# or default.qubit
dev = qml.device("lightning.gpu", wires=4)

# Random circuit parameters
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))

@qml.qnode(dev)
def circuit(phi):
    for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)

    # Random quantum circuit
    RandomLayers(rand_params, wires=list(range(4)))

    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

# Quantum convolution for each channel
def quanv(image):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((32, 32, 12))  # process 3 channels separately, producing 12-channel output

    for j in range(0, 64, 2):
        for k in range(0, 64, 2):
            for ch in range(3):  # RGB
                q_results = circuit(
                    [
                        image[j, k, ch],
                        image[j + 1, k, ch],
                        image[j, k + 1, ch],
                        image[j + 1, k + 1, ch],
                    ]
                )
                for c in range(4):
                    out[j // 2, k // 2, ch * 4 + c] = q_results[c]  
    return out

if PREPROCESS:
    q_train_images = []
    print("Quantum pre-processing of train images:")
    for idx, img in enumerate(train_images):
        print(f"{idx + 1}/{n_train}", end="\r")
        q_train_images.append(quanv(img.cpu().numpy())) 
    q_train_images = np.asarray(q_train_images)

    np.save(SAVE_PATH + "/eos_0.3to0.4.npy", q_train_images)

print("Data process time: ", time.time()-start)