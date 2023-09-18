from PIL import ImageOps


def apply_augmentations(img: PImage.Image) -> PImage.Image:
    # 颜色抖动
    if np.random.rand() < 0.5:  # 50%的机会应用颜色抖动
        img = color_jitter(img)

    # 垂直翻转
    if np.random.rand() < 0.5:  # 50%的机会应用垂直翻转
        img = vertical_flip(img)

    # 水平翻转
    if np.random.rand() < 0.5:  # 50%的机会应用水平翻转
        img = horizontal_flip(img)

    return img


def save_image(record, directory, is_training=False):
    image_id = record['id']
    dst = os.path.join(directory, image_id + '.png')
    resp = requests.get(API_BASE_URL + f'image/{image_id}')
    if resp.status_code != 200:
        raise ValueError(f'Failed to get image with ID {image_id}. Response: {resp.text}')

    in_stream = io.BytesIO(resp.content)
    pimage = PImage.open(in_stream)

    # 如果是训练数据，应用数据增强
    if is_training:
        pimage = apply_augmentations(pimage)

    # Crop the image by intensity percentile
    pimage = crop_by_percentile(pimage)

    # 保存增强后的图像
    pimage.save(dst)


# 修改handle_category以标记哪些是训练数据
def handle_category(category, records):
    np.random.shuffle(records)
    n = len(records)
    n_train = int(0.8 * n)
    n_dev = int(0.9 * n) - n_train

    for record in records[:n_train]:
        save_image(record, os.path.join(train_dir, category), is_training=True)  # 标记为训练数据
    for record in records[n_train:n_train + n_dev]:
        save_image(record, os.path.join(dev_dir, category))
    for record in records[n_train + n_dev:]:
        save_image(record, os.path.join(test_dir, category))
