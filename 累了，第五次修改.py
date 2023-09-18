def handle_augmented_category(category, records):
    np.random.shuffle(records)
    n = len(records)
    n_train = int(0.8 * n)
    n_dev = int(0.9 * n) - n_train

    # Only augment training data
    for record in records[:n_train]:
        augment_and_save_image(record, os.path.join(train_dir, category))

    # Save dev and test data without augmentation
    for record in records[n_train:n_train + n_dev]:
        save_image(record, os.path.join(dev_dir, category))
    for record in records[n_train + n_dev:]:
        save_image(record, os.path.join(test_dir, category))

# Use ThreadPoolExecutor to parallelize the saving of images
with ThreadPoolExecutor(max_workers=10) as executor:
    for category, records in records_by_category.items():
        executor.submit(handle_augmented_category, category, records)
