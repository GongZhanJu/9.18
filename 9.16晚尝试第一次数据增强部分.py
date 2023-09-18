import os
import numpy as np
import requests
import io
from PIL import Image as PImage, ImageEnhance, ImageOps
from concurrent.futures import ThreadPoolExecutor
import pathlib

# Configurations
API_BASE_URL = 'http://fireeye-test-backend-container:9090/api/'
TF_SERVING_BASE_URL = 'http://fireeye-test-model-container:8501/'
task_id = '1ac1e8a095df4611af387d9934799251'
image_dir = './ima'
id_code_mapping = {
    'dbee3deebc5444f5b011da4e5518752c': '0',
    'edb4cb51d54644c08aa122d3f041bb0a': '1'
}

task_dir = os.path.join(image_dir, task_id)
train_dir = os.path.join(task_dir, 'train')
dev_dir = os.path.join(task_dir, 'dev')
test_dir = os.path.join(task_dir, 'test')


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
    return img_array / 255.0


def color_jitter(img: PImage.Image, brightness=0.2, contrast=0.2, saturation=0.2) -> PImage.Image:
    img = ImageEnhance.Brightness(img).enhance(1 + brightness * (2 * np.random.random() - 1))
    img = ImageEnhance.Contrast(img).enhance(1 + contrast * (2 * np.random.random() - 1))
    img = ImageEnhance.Color(img).enhance(1 + saturation * (2 * np.random.random() - 1))
    return img


def vertical_flip(img: PImage.Image) -> PImage.Image:
    return ImageOps.flip(img)


def horizontal_flip(img: PImage.Image) -> PImage.Image:
    return ImageOps.mirror(img)


def save_image(record, directory):
    image_id = record['id']
    dst = os.path.join(directory, image_id + '.png')
    resp = requests.get(API_BASE_URL + f'image/{image_id}')
    if resp.status_code != 200:
        raise ValueError(f'Failed to get image with ID {image_id}. Response: {resp.text}')
    in_stream = io.BytesIO(resp.content)
    pimage = PImage.open(in_stream)

    cropped_img = crop_by_percentile(pimage)
    normalized_img_array = normalize_image(cropped_img)
    pimage = PImage.fromarray((normalized_img_array * 255).astype(np.uint8))
    pimage.save(dst)


def handle_category(category, records):
    np.random.shuffle(records)
    n = len(records)
    n_train = int(0.8 * n)
    n_dev = int(0.9 * n) - n_train

    for record in records[:n_train]:
        save_image(record, os.path.join(train_dir, category))

        image_id = record['id']
        original_image_path = os.path.join(train_dir, category, image_id + '.png')
        original_image = PImage.open(original_image_path)

        jittered_img = color_jitter(original_image)
        jittered_img_path = os.path.join(train_dir, category, f"{image_id}_jittered.png")
        jittered_img.save(jittered_img_path)

        flipped_v_img = vertical_flip(original_image)
        flipped_v_img_path = os.path.join(train_dir, category, f"{image_id}_flipped_v.png")
        flipped_v_img.save(flipped_v_img_path)

        flipped_h_img = horizontal_flip(original_image)
        flipped_h_img_path = os.path.join(train_dir, category, f"{image_id}_flipped_h.png")
        flipped_h_img.save(flipped_h_img_path)

    for record in records[n_train:n_train + n_dev]:
        save_image(record, os.path.join(dev_dir, category))
    for record in records[n_train + n_dev:]:
        save_image(record, os.path.join(test_dir, category))


# Creating directories
for category in records_by_category.keys():
    for directory in [train_dir, dev_dir, test_dir]:
        os.makedirs(os.path.join(directory, category), exist_ok=True)

# Saving images
with ThreadPoolExecutor(max_workers=10) as executor:
    for category, records in records_by_category.items():
        executor.submit(handle_category, category, records)

train_image_count = len(list(pathlib.Path(train_dir).glob('*/*.png')))
dev_image_count = len(list(pathlib.Path(dev_dir).glob('*/*.png')))
test_image_count = len(list(pathlib.Path(test_dir).glob('*/*.png')))
print(train_image_count, dev_image_count, test_image_count)
