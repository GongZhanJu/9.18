def augment_and_save_image(record, directory):
    image_id = record['id']
    resp = requests.get(API_BASE_URL + f'image/{image_id}')
    if resp.status_code != 200:
        raise ValueError(f'Failed to get image with ID {image_id}. Response: {resp.text}')

    in_stream = io.BytesIO(resp.content)
    pimage = PImage.open(in_stream)

    # 保存原始图像
    pimage.save(os.path.join(directory, image_id + '_original.png'))

    # 应用颜色抖动并保存
    img_jittered = color_jitter(pimage)
    img_jittered.save(os.path.join(directory, image_id + '_jittered.png'))

    # 应用垂直翻转并保存
    img_vflipped = vertical_flip(pimage)
    img_vflipped.save(os.path.join(directory, image_id + '_vflipped.png'))

    # 应用水平翻转并保存
    img_hflipped = horizontal_flip(pimage)
    img_hflipped.save(os.path.join(directory, image_id + '_hflipped.png'))


def handle_augmented_category(category, records):
    np.random.shuffle(records)
    n = len(records)
    n_train = int(0.8 * n)

    for record in records[:n_train]:
        augment_and_save_image(record, os.path.join(train_dir, category))
