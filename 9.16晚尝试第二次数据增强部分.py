# ... [previous code including function definitions]

def download_image(record, directory):
    # This function will ONLY download and save the image without any processing
    image_id = record['id']
    dst = os.path.join(directory, image_id + '.png')
    resp = requests.get(API_BASE_URL + f'image/{image_id}')
    if resp.status_code != 200:
        raise ValueError(f'Failed to get image with ID {image_id}. Response: {resp.text}')
    in_stream = io.BytesIO(resp.content)
    pimage = PImage.open(in_stream)
    pimage.save(dst)


def crop_normalize_and_save_images(directory, category):
    # This function will read images, crop, normalize them and overwrite the original ones
    for image_name in os.listdir(os.path.join(directory, category)):
        img_path = os.path.join(directory, category, image_name)
        pimage = PImage.open(img_path)

        cropped_img = crop_by_percentile(pimage)
        normalized_img_array = normalize_image(cropped_img)
        normalized_img = PImage.fromarray((normalized_img_array * 255).astype(np.uint8))

        # Overwrite the original image
        normalized_img.save(img_path)


def augment_and_save_training_images(directory, category):
    # This function will apply data augmentation on training images
    for image_name in os.listdir(os.path.join(directory, category)):
        img_path = os.path.join(directory, category, image_name)
        pimage = PImage.open(img_path)

        jittered_img = color_jitter(pimage)
        flipped_v_img = vertical_flip(pimage)
        flipped_h_img = horizontal_flip(pimage)

        # Save the additional processed images
        jittered_img.save(img_path.replace(".png", "_jittered.png"))
        flipped_v_img.save(img_path.replace(".png", "_flipped_v.png"))
        flipped_h_img.save(img_path.replace(".png", "_flipped_h.png"))


# Downloading images without processing
with ThreadPoolExecutor(max_workers=10) as executor:
    for category, records in records_by_category.items():
        executor.submit(download_image, category, records)

# Cropping and Normalizing the images without data augmentation
for category in records_by_category.keys():
    for directory in [train_dir, dev_dir, test_dir]:
        crop_normalize_and_save_images(directory, category)

# Applying data augmentation only to the training set
for category in records_by_category.keys():
    augment_and_save_training_images(train_dir, category)

# ... [rest of the code]
