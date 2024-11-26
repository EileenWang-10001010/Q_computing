The score comes from inference task of grounding DINO (grounding-dino-base).


File: score_with_grounding_DINO.py
Object detection inference task with DINO:
In each inference, you provide image + text prompt. Here, the text prompt is "round dark."
You may set thresholds (box_threshold and text_threshold) to ignore low quality images.

Result:
You can get image name with its score. The score is the confidence level of the bounding box given by DINO.
The result example is select_lym_0.2.txt

File: Select_img_name_list_given_threshold.ipynb

Once you have the quality score of images (e.g. select_lym_0.2.txt), you can set a threshold and get the name list of these images.