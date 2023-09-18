import os
import numpy as np
from PIL import Image as PImage, ImageEnhance, ImageOps
import requests
import io
from concurrent.futures import ThreadPoolExecutor

API_BASE_URL = 'http://fireeye-test-backend-container:9090/api/'
image_dir = './ima'
task_id = '1ac1e8a095df4611af387d9934799251'
id_code_mapping = {
    'dbee3deebc5444f5b011da4e5518752c': '0',
    'edb4cb51d54644c08aa122d3f041bb0a': '1'
}

task_dir = os.path.join(image_dir, task_id)
train_dir = os.path.join(task_dir, 'train')
dev_dir = os.path.join(task_dir, 'dev')
test_dir = os.path.join(task_dir, 'test')


def save_image(record, directory):
    image_id = record['id']
    dst = os.path.join(directory, image_id + '.png')
    resp = requests.get(API_BASE_URL + f'image/{image_id}')
    if resp.status_code != 200:
        raise ValueError(f'Failed to get image with ID {image_id}. Response: {resp.text}')
    in_stream = io.BytesIO(resp.content)
    pimage = PImage.open(in_stream)
    pimage.save(dst)


def crop_by_percentile(img, lower_percentile=5, upper_percentile=95):
    img_array = np.array(img.convert('L'))
    low_val, high_val = np.percentile(img_array, [lower_percentile, upper_percentile])
    mask = np.logical_and(img_array > low_val, img_array < high_val)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped_img = img.crop((cmin, rmin, cmax, rmax))
    return cropped_img


def normalize_image(img: PImage.Image) -> np.ndarray:
    img_array = np.array(img)
    normalized = img_array / 255.0
    return PImage.fromarray((normalized * 255).astype(np.uint8))


def augment_training_images(img_dir):
    filenames = os.listdir(img_dir)
    for filename in filenames:
        if not filename.endswith('.png'):
            continue

        img_path = os.path.join(img_dir, filename)
        img = PImage.open(img_path)

        # Data Augmentation
        img = color_jitter(img)
        img = vertical_flip(img)
        img = horizontal_flip(img)

        # Save Augmented Images
        augmented_path = os.path.join(img_dir, "augmented_" + filename)
        img.save(augmented_path)


# Crop and Normalize
directories = [train_dir, dev_dir, test_dir]
for directory in directories:
    filenames = os.listdir(directory)
    for filename in filenames:
        if not filename.endswith('.png'):
            continue

        img_path = os.path.join(directory, filename)
        img = PImage.open(img_path)
        img = crop_by_percentile(img)
        img = normalize_image(img)
        img.save(img_path)

# Data Augmentation on training set
augment_training_images(train_dir)
