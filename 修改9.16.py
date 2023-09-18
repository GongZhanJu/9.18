import os
import io
import numpy as np
import requests
from PIL import Image as PImage, ImageEnhance, ImageOps
from concurrent.futures import ThreadPoolExecutor

# ... [your functions like crop_by_percentile, normalize_image, etc.] ...

# Directories setup
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


def preprocess_image(image_stream):
    img = PImage.open(image_stream)
    cropped_img = crop_by_percentile(img)
    normalized_img_array = normalize_image(cropped_img)
    return PImage.fromarray((normalized_img_array * 255).astype(np.uint8))


def save_image(record, directory, augment=False):
    image_id = record['id']
    resp = requests.get(API_BASE_URL + f'image/{image_id}')
    if resp.status_code != 200:
        raise ValueError(f'Failed to get image with ID {image_id}. Response: {resp.text}')

    # Prune and normalize
    img_stream = io.BytesIO(resp.content)
    img = preprocess_image(img_stream)
    img_path = os.path.join(directory, f'{image_id}.png')
    img.save(img_path)

    # Data augmentation (if augment=True and it's the training directory)
    if augment:
        color_jitter(img).save(os.path.join(directory, f'{image_id}_colorjittered.png'))
        vertical_flip(img).save(os.path.join(directory, f'{image_id}_vflipped.png'))
        horizontal_flip(img).save(os.path.join(directory, f'{image_id}_hflipped.png'))


def handle_category(category, records):
    np.random.shuffle(records)
    n = len(records)
    n_train = int(0.8 * n)
    n_dev = int(0.9 * n) - n_train

    for record in records[:n_train]:
        save_image(record, os.path.join(train_dir, category), augment=True)
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
