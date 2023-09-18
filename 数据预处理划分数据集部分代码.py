import json
import os
import requests
import io
import shutil
import math
import PIL
import numpy as np
import glob
import shutil
import PIL.Image as PImage
from PIL import ImageEnhance
from pprint import pprint
from collections import Counter
from datetime import datetime
from PIL import Image, ImageOps, ImageEnhance
from sklearn.model_selection import train_test_split


API_BASE_URL = 'http://fireeye-test-backend-container:9090/api/'
TF_SERVING_BASE_URL = 'http://fireeye-test-model-container:8501/'
task_id='1ac1e8a095df4611af387d9934799251'
id_code_mapping = {
    'dbee3deebc5444f5b011da4e5518752c': '0',
    'edb4cb51d54644c08aa122d3f041bb0a': '1'}


num_images = requests.get(
    url=API_BASE_URL+'image/count',
    params=dict(
        task_id=task_id,
        has_truth=True
    )
).json()

print('该图片数量：',num_images)




def get_image_by_id(image_id):
    """Retrieve image by its ID."""
    r = requests.get(url=API_BASE_URL + 'image/' + image_id)
    if r.status_code == 200:
        return PImage.open(io.BytesIO(r.content))
    else:
        raise RuntimeError(r.text)
#img = get_image_by_id(image_records[200]['id'])
#img.show()


import pprint
def get_image_records(task_id):
    """Fetch image records given a task ID."""
    resp = requests.get(
        url=API_BASE_URL + 'image',
        params={'task_id': task_id, 'has_truth': True}
    )
    if resp.status_code == 200:
        return resp.json()
    else:
        raise RuntimeError(resp.text)
image_records = get_image_records(task_id)


def crop_white_border(img, threshold=240):
    """Crop white borders from an image."""
    img_array = np.array(img)
    non_white_rows = np.any(img_array < threshold, axis=(1, 2))
    non_white_columns = np.any(img_array < threshold, axis=(0, 2))
    row_min, row_max = np.where(non_white_rows)[0][[0, -1]]
    col_min, col_max = np.where(non_white_columns)[0][[0, -1]]
    cropped_img = img.crop((col_min, row_min, col_max, row_max))
    return cropped_img


def normalize_image(img: Image.Image) -> np.ndarray:
    img_array = np.array(img)
    return img_array / 255.0


image_records = get_image_records(task_id)


image_dir = "./images"
category0_dir = os.path.join(image_dir, 'Category0')
category1_dir = os.path.join(image_dir, 'Category1')
if not os.path.exists(category0_dir):
    os.makedirs(category0_dir)

if not os.path.exists(category1_dir):
    os.makedirs(category1_dir)


for record in image_records:
    try:
        img = get_image_by_id(record['id'])
        cropped_img = crop_white_border(img)
        normalized_img_array = normalize_image(cropped_img)
        normalized_img = PImage.fromarray((normalized_img_array * 255).astype(np.uint8))
        truth_id = record['truth_id']
        if id_code_mapping[truth_id] == '0':
            file_path = os.path.join(category0_dir, f"{record['id']}.png")
        else:
            file_path = os.path.join(category1_dir, f"{record['id']}.png")
        normalized_img.save(file_path, 'PNG')
    except Exception as e:
        print(f'Error processing image {record["id"]}. Error: {e}')


def download_image(image_id):
    response = requests.get(f"{API_BASE_URL}image/download/{image_id}")
    return response.content


def crop_white_border(img, threshold=240):
    img_array = np.array(img)
    non_white_rows = np.any(img_array < threshold, axis=(1, 2))
    non_white_columns = np.any(img_array < threshold, axis=(0, 2))
    row_min, row_max = np.where(non_white_rows)[0][[0, -1]]
    col_min, col_max = np.where(non_white_columns)[0][[0, -1]]
    cropped_img = img.crop((col_min, row_min, col_max, row_max))
    return cropped_img


def vertical_flip(img: Image.Image) -> Image.Image:
    return ImageOps.flip(img)


def horizontal_flip(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)


import shutil


train_dir = './train_images'
test_dir = './test_images'
val_dir = './val_images'

for dir_path in [train_dir, test_dir, val_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


all_images = [os.path.join(category0_dir, f'{record["id"]}.png') for record in image_records if id_code_mapping[record["truth_id"]] == "0"] +\
             [os.path.join(category1_dir, f'{record["id"]}.png') for record in image_records if id_code_mapping[record["truth_id"]] == "1"]

labels = [id_code_mapping[record['truth_id']] for record in image_records]

train_images, test_images, train_labels, test_labels= train_test_split(all_images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

for img_path in train_images:
    shutil.move(img_path, train_dir)
for img_path in test_images:
    shutil.move(img_path, test_dir)
for img_path in val_images:
    shutil.move(img_path, val_dir)


import os
from PIL import Image


train_dir = './test_images/'

for filename in os.listdir(train_dir):
    if filename.endswith(".png"):
        img_path = os.path.join(train_dir, filename)
        img = Image.open(img_path)

        jittered_img = color_jitter(img)
        jittered_filename = "jittered_" + filename
        jittered_img.save(os.path.join(train_dir, jittered_filename))

        v_flipped_img = vertical_flip(img)
        vflipped_filename = "vflipped_" + filename
        v_flipped_img.save(os.path.join(train_dir, vflipped_filename))

        h_flipped_img = horizontal_flip(img)
        hflipped_fliename = "hflipped_" + filename
        h_flipped_img.save(os.path.join(train_dir, hflipped_fliename))

print('Data augmentation for teh training set is complete.')

