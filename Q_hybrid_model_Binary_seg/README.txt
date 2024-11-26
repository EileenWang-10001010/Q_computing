Binary segmentation with hybrid models

A very simple integration of quantum-classical hybrid model, with only a quantum layer in the middle between encoder and decoder.

Quantum circuit design is from pennylane tutorial:
https://github.com/PennyLaneAI/qml/blob/master/demonstrations/tutorial_qnn_module_torch.py

Dataset:
images.npy: (n, 64, 64, 3)
masks.npy: (n, 64, 64, 1)