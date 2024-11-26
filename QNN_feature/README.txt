Can see the one-channel image quantum feature extraction tutorial from:
https://pennylane.ai/qml/demos/tutorial_quanvolution

Originally, quantum feature generation + training are in the same file.
But for better readability, I split them into two files.

File: q_feature_generator.py
Description:
image (64, 64, 3) -> quantum feature (32, 32, 12)
Using 2x2 Q kernel to extract features, with 1 random quantum layer.
Result quantum feature stacks are in result_q_feature/....npy, the original images are from TCGA_cell_image_extraction_64px/Eosinophil and TCGA_cell_image_extraction_64px/Lymphocyte

For comparison, the sequence of quantum feature stack is in result_q_feature/filenameList/....txt