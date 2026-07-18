import os
import shutil
import random

# Path of original YOLO dataset
source_path = "original_dataset"

# Path where client datasets will be created
destination_path = "dataset"

clients = ["client1", "client2", "client3"]
splits = ["train", "valid", "test"]

for split in splits:

    image_dir = os.path.join(source_path, split, "images")
    label_dir = os.path.join(source_path, split, "labels")

    images = os.listdir(image_dir)

    random.shuffle(images)

    split_size = len(images) // 3

    client_data = {
        "client1": images[:split_size],
        "client2": images[split_size:2 * split_size],
        "client3": images[2 * split_size:]
    }

    for client in clients:

        client_image_dir = os.path.join(destination_path, client, split, "images")
        client_label_dir = os.path.join(destination_path, client, split, "labels")

        os.makedirs(client_image_dir, exist_ok=True)
        os.makedirs(client_label_dir, exist_ok=True)

        for image_file in client_data[client]:

            src_image = os.path.join(image_dir, image_file)
            dst_image = os.path.join(client_image_dir, image_file)

            label_file = os.path.splitext(image_file)[0] + ".txt"

            src_label = os.path.join(label_dir, label_file)
            dst_label = os.path.join(client_label_dir, label_file)

            shutil.copy(src_image, dst_image)

            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)

print("Dataset successfully split into clients!")