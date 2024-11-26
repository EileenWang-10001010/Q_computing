import os
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

PATH = os.getcwd()
lst = ["Eosinophil", "Lymphocyte", "Neutrophil", "Plasmacell", "Tumor"]
img = []

sample_n = 300
threshold = 127
mask_stack = []
img_stack = []

mode = "RGB_fold01" #"Bi" #"RGB"

# Define the RGB values for each label
label_colors = [
        [0, 0, 255],  #lym
        [255, 255, 0],  #tum
        [0, 255, 0],  #neu
        [255, 0, 0],  #plas
        [255, 0, 255], #eos
]

# Convert to NumPy array for easier comparison
label_colors = np.array(label_colors)

os.makedirs(f"sampleList/sample_{sample_n}_{mode}_mask/all", exist_ok=True)

for classname in lst:
    filenames = os.listdir(f"{PATH}/{classname}/")
    sample = np.random.choice(filenames, size=sample_n, replace=False).tolist()
    # if len(filenames) > 500:  
    #     sample = np.random.choice(filenames, size=len(filenames)//2, replace=False).tolist()
    # else:
    #     sample = filenames

    os.makedirs(f"sampleList/sample_{sample_n}_{mode}_mask/all/{classname}", exist_ok=True)
    
    for img in sample:
        im = cv2.imread(f"{PATH}/{classname}/{img}")
        cv2.imwrite(f"sampleList/sample_{sample_n}_{mode}_mask/all/{classname}/{img}", im)
        im = np.array(im)
        seg_origin = im[:,:64,:]
        # binary_masks = np.expand_dims(np.where(im[:,128:,0]>127,1, 0).astype(np.uint8), axis=-1)

        
        seg_mask = im[:,64:128,:]
        seg_mask = np.where(seg_mask > threshold, 255, 0).astype(np.uint8)

        binary_masks = np.zeros((64, 64, 6), dtype=np.uint8)
        for i, color in enumerate(label_colors):
            binary_masks[:,:,i] = np.all(seg_mask == color, axis=-1).astype(np.uint8)
        # binary_masks[:,:,5] = np.mean(im[:,128:,:], axis=-1).astype(np.uint8)
        binary_masks[:,:,5] = (np.where(im[:,128:,0]>127,1, 0)).astype(np.uint8)
        
        # background_mask = 1 - np.clip(np.sum(binary_masks, axis=-1), 0, 1)  # Shape: (64, 64)

        # Step 2: Stack the 5 masks with the background mask
        # final_mask = np.concatenate([binary_masks, background_mask[..., np.newaxis]], axis=-1)

        # print(classname, seg_mask[32, 32, :])

        img_stack.append(seg_origin)
        mask_stack.append(binary_masks)


    os.makedirs(f"sampleList/sample_{sample_n}_{mode}_mask/name", exist_ok=True)
    np.savetxt(f'sampleList/sample_{sample_n}_{mode}_mask/name/{classname}_sample_{sample_n}.txt', sample, fmt='%s')
    
    # os.makedirs(f"sampleList/sample_{sample_n}_{mode}_mask/image", exist_ok=True)
    # np.save(f"sampleList/sample_{sample_n}_{mode}_mask/image/image_{classname}.npy",img_stack)
    # os.makedirs(f"sampleList/sample_{sample_n}_{mode}_mask/mask", exist_ok=True)
    # np.save(f"sampleList/sample_{sample_n}_{mode}_mask/mask/mask_{classname}.npy",mask_stack)

indices = np.arange(len(img_stack))
X_train, X_val, y_train, y_val, train_indices, test_indices = train_test_split(img_stack, mask_stack, indices, test_size=0.2, random_state=7)

os.makedirs(f"sampleList/sample_{sample_n}_{mode}_mask/fold0", exist_ok=True)
os.makedirs(f"sampleList/sample_{sample_n}_{mode}_mask/fold1", exist_ok=True)

np.save(f"sampleList/sample_{sample_n}_{mode}_mask/fold0/images.npy",X_train)
np.save(f"sampleList/sample_{sample_n}_{mode}_mask/fold0/masks.npy",y_train)
np.savetxt(f"sampleList/sample_{sample_n}_{mode}_mask/fold0/index.txt", train_indices, fmt = '%d')

np.save(f"sampleList/sample_{sample_n}_{mode}_mask/fold1/images.npy",X_val)
np.save(f"sampleList/sample_{sample_n}_{mode}_mask/fold1/masks.npy",y_val)
np.savetxt(f"sampleList/sample_{sample_n}_{mode}_mask/fold1/index.txt", test_indices, fmt = '%d')
# np.save(f"sampleList/sample_{sample_n}_{mode}_mask/images.npy",img_stack)
# np.save(f"sampleList/sample_{sample_n}_{mode}_mask/masks.npy",mask_stack)

print(np.array(img_stack).shape, np.array(mask_stack).shape)

