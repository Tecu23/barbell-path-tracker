from pathlib import Path

import cv2
import numpy as np


class DataLoader:
    def __init__(self, dataset_path, input_size=(128, 128)):
        self.dataset_path = Path(dataset_path)
        self.input_size = input_size

        # Define paths
        self.train_images = self.dataset_path / "train" / "images"
        self.train_labels = self.dataset_path / "train" / "labels"
        self.val_images = self.dataset_path / "valid" / "images"
        self.val_labels = self.dataset_path / "valid" / "labels"

    def parse_yolo_label(self, label_path):
        """Parse YOLO format label file"""
        if not label_path.exists():
            return [0, 0, 0, 0], 0.0  # No barbell found

        with open(label_path, "r") as f:
            lines = f.readlines()

        if not lines:
            return [0, 0, 0, 0], 0.0

        # Parse first line (assuming single barbell per image)
        parts = lines[0].strip().split()
        if len(parts) >= 5:
            # class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            return [x_center, y_center, width, height], 1.0

        return [0, 0, 0, 0], 0.0

    def load_image(self, image_path):
        """Load and preprocess image"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img_normalized = img_resized.astype(np.float32) / 255.0

        return img_normalized

    def load_dataset(self, split="train"):
        """Load training or validation dataset"""
        if split == "train":
            images_dir = self.train_images
            labels_dir = self.train_labels
        else:
            images_dir = self.val_images
            labels_dir = self.val_labels

        images = []
        bboxes = []
        confidences = []

        # Get all image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        print(f"Found {len(image_files)} {split} images")

        for img_path in image_files:
            try:
                # Load image
                img = self.load_image(img_path)

                # Load corresponding label
                label_path = labels_dir / f"{img_path.stem}.txt"
                bbox, conf = self.parse_yolo_label(label_path)

                images.append(img)
                bboxes.append(bbox)
                confidences.append(conf)

            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

        return np.array(images), np.array(bboxes), np.array(confidences)
