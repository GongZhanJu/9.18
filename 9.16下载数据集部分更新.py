import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Directory Initialization
task_dir = os.path.join(image_dir, task_id)
train_dir = os.path.join(task_dir, 'train')
dev_dir = os.path.join(task_dir, 'dev')
test_dir = os.path.join(task_dir, 'test')

# Split images stratified by their categories
records_by_category = {}
for record in image_records:
    category = id_code_mapping.get(str(record['truth_id']), None)
    if category is not None:
        records_by_category.setdefault(category, []).append(record)

def save_image(record, directory):
    image_id = record['id']
    dst = os.path.join(directory, image_id + '.png')
    resp = requests.get(API_BASE_URL + f'image/{image_id}')
    if resp.status_code != 200:
        raise ValueError(f'Failed to get image with ID {image_id}. Response: {resp.text}')
    in_stream = io.BytesIO(resp.content)
    pimage = PImage.open(in_stream)
    pimage.save(dst)

def handle_category(category, records):
    np.random.shuffle(records)
    n = len(records)
    n_train = int(0.8 * n)
    n_dev = int(0.9 * n) - n_train

    for record in records[:n_train]:
        save_image(record, os.path.join(train_dir, category))
    for record in records[n_train:n_train + n_dev]:
        save_image(record, os.path.join(dev_dir, category))
    for record in records[n_train + n_dev:]:
        save_image(record, os.path.join(test_dir, category))

# Ensure directories exist
for category in records_by_category.keys():
    for directory in [train_dir, dev_dir, test_dir]:
        os.makedirs(os.path.join(directory, category), exist_ok=True)

# Use ThreadPoolExecutor to parallelize the saving of images
with ThreadPoolExecutor(max_workers=10) as executor:
    for category, records in records_by_category.items():
        executor.submit(handle_category, category, records)


import pathlib

train_image_count = len(list(pathlib.Path(train_dir).glob('*/*.png')))
dev_image_count = len(list(pathlib.Path(dev_dir).glob('*/*.png')))
test_image_count = len(list(pathlib.Path(test_dir).glob('*/*.png')))
print(train_image_count, dev_image_count, test_image_count)