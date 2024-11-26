# import cv2
import os
import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

start = time.time()

# Your path to images
path = "/home/tester/Ai_Lin/"
lym = os.listdir(path+'Lymphocyte')

model_id = "IDEA-Research/grounding-dino-base"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

lst = []
score = []

for i, cell in enumerate(lym):
        
        image = Image.open(f"{path}Lymphocyte/{cell}") # PATH to images

    # VERY important: text queries need to be lowercased + end with a dot
        text = "round dark."

        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.2, # can tune
            text_threshold=0.3, # can  tune
            target_sizes=[image.size[::-1]]
        )
        
        if results[0]["labels"]: 
            lst.append(cell)
            score.append(results[0]["scores"][0])
            # image = image.save(f"{c}_lym.jpg")

with open('select_lym_0.2.txt', 'w') as f:
    for name, confidence in zip(lst, score):
        f.write(f"{name}  {confidence}\n")

print(len(lst))
print("time span: ", time.time() - start)
