def save_image(record, directory):
    image_id = record['id']
    dst = os.path.join(directory, image_id + '.png')
    resp = requests.get(API_BASE_URL + f'image/{image_id}')
    if resp.status_code != 200:
        raise ValueError(f'Failed to get image with ID {image_id}. Response: {resp.text}')

    in_stream = io.BytesIO(resp.content)
    pimage = PImage.open(in_stream)

    # Cropping and normalization
    cropped_img = crop_by_percentile(pimage)
    normalized_img_array = normalize_image(cropped_img)

    # Convert the normalized array back to an Image object
    processed_image = PImage.fromarray((normalized_img_array * 255).astype(np.uint8))
    processed_image.save(dst)
