# ... [the beginning part of your code remains unchanged]

train_records, test_records, train_labels, test_labels = train_test_split(
    image_records, labels, test_size=0.3, stratify=labels, random_state=42)

train_records, val_records, train_labels, val_labels = train_test_split(
    train_records, train_labels, test_size=0.1, stratify=train_labels, random_state=42)

# Processing & Saving for Training Set
for record in train_records:
    try:
        img = get_image_by_id(record['id'])
        cropped_img = crop_by_percentile(img)
        normalized_img_array = np.array(cropped_img) / 255.0
        normalized_img = Image.fromarray((normalized_img_array * 255).astype(np.uint8))

        truth_id = record['truth_id']
        category = id_code_mapping[truth_id]

        directory = os.path.join(base_dir, 'train', f'Category{category}')
        file_path = os.path.join(directory, f'{record["id"]}.png')
        normalized_img.save(file_path, 'PNG')
    except Exception as e:
        print(f'Error processing image {record["id"]}. Error: {e}')

# Saving images for Test and Validation Sets without processing
for set_name, records in [('test', test_records), ('val', val_records)]:
    for record in records:
        try:
            img = get_image_by_id(record['id'])

            truth_id = record['truth_id']
            category = id_code_mapping[truth_id]

            directory = os.path.join(base_dir, set_name, f'Category{category}')
            file_path = os.path.join(directory, f'{record["id"]}.png')
            img.save(file_path, 'PNG')
        except Exception as e:
            print(f'Error saving image {record["id"]}. Error: {e}')
