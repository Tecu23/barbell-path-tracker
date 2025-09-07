import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_yolo_annotation(label_path):
    """Load YOLO format annotation from .txt file"""
    if not label_path.exists():
        return None

    with open(label_path, "r") as f:
        line = f.readline().strip()

    if not line:
        return None

    parts = line.split()
    if len(parts) < 5:
        return None

    # YOLO format: class_id x_center y_center width height (all normalized 0-1)
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])

    return {
        "class_id": class_id,
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
    }


def draw_bbox_on_image(image, annotation):
    """Draw bounding box on image using YOLO annotation"""
    if annotation is None:
        return image

    img_height, img_width = image.shape[:2]

    # Convert normalized coordinates to pixel coordinates
    x_center = annotation["x_center"] * img_width
    y_center = annotation["y_center"] * img_height
    bbox_width = annotation["width"] * img_width
    bbox_height = annotation["height"] * img_height

    # Calculate corner coordinates
    x1 = int(x_center - bbox_width / 2)
    y1 = int(y_center - bbox_height / 2)
    x2 = int(x_center + bbox_width / 2)
    y2 = int(y_center + bbox_height / 2)

    # Draw bounding box (red rectangle)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw center point (green circle)
    cv2.circle(image, (int(x_center), int(y_center)), 5, (0, 255, 0), -1)

    # Add label text
    label_text = f"Barbell"
    cv2.putText(
        image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )

    # Add coordinates text
    coord_text = f"Center: ({int(x_center)}, {int(y_center)})"
    cv2.putText(
        image, coord_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
    )

    return image


def view_single_sample(images_dir, labels_dir, image_name):
    """View a single training sample with annotation"""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    # Load image
    image_path = images_dir / image_name
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load annotation
    label_path = labels_dir / f"{image_path.stem}.txt"
    annotation = load_yolo_annotation(label_path)

    # Draw bounding box
    image_with_bbox = draw_bbox_on_image(image_rgb.copy(), annotation)

    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(image_with_bbox)
    plt.title(f"Training Sample: {image_name}")
    plt.axis("off")

    # Print annotation info
    if annotation:
        print(f"Image: {image_name}")
        print(f"Annotation found:")
        print(f"  Class ID: {annotation['class_id']}")
        print(
            f"  Center: ({annotation['x_center']:.3f}, {annotation['y_center']:.3f}) [normalized]"
        )
        print(
            f"  Size: {annotation['width']:.3f} x {annotation['height']:.3f} [normalized]"
        )
    else:
        print(f"No annotation found for {image_name}")

    plt.show()


def view_random_samples(dataset_path, split="train", num_samples=6):
    """View multiple random training samples in a grid"""
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / split / "images"
    labels_dir = dataset_path / split / "labels"

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return

    # Find all image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    # Select random samples
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))

    # Calculate grid layout
    cols = 3
    rows = (len(selected_files) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif len(selected_files) == 1:
        axes = np.array([[axes]])

    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        if i < len(selected_files):
            image_path = selected_files[i]

            # Load image
            image = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load annotation
            label_path = labels_dir / f"{image_path.stem}.txt"
            annotation = load_yolo_annotation(label_path)

            # Draw bounding box
            image_with_bbox = draw_bbox_on_image(image_rgb.copy(), annotation)

            # Display
            ax.imshow(image_with_bbox)

            # Title with annotation info
            if annotation:
                title = f"{image_path.name}\n✓ Annotated"
                ax.set_title(title, fontsize=10, color="green")
            else:
                title = f"{image_path.name}\n✗ No annotation"
                ax.set_title(title, fontsize=10, color="red")
        else:
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def browse_dataset(dataset_path, split="train"):
    """Interactive browsing of dataset samples"""
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / split / "images"
    labels_dir = dataset_path / split / "labels"

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return

    # Find all image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(image_files)} images in {split} set")
    print("Commands:")
    print("  - Enter image index (0 to {})".format(len(image_files) - 1))
    print("  - 'r' for random image")
    print("  - 'q' to quit")
    print("  - 'list' to show all filenames")

    while True:
        command = (
            input(f"\nEnter command (0-{len(image_files)-1}, 'r', 'list', 'q'): ")
            .strip()
            .lower()
        )

        if command == "q":
            break
        elif command == "r":
            # Show random image
            selected_file = random.choice(image_files)
            view_single_sample(images_dir, labels_dir, selected_file.name)
        elif command == "list":
            # List all files
            for i, file in enumerate(image_files):
                print(f"{i:3d}: {file.name}")
        elif command.isdigit():
            index = int(command)
            if 0 <= index < len(image_files):
                selected_file = image_files[index]
                view_single_sample(images_dir, labels_dir, selected_file.name)
            else:
                print(f"Invalid index. Use 0 to {len(image_files)-1}")
        else:
            print(
                "Invalid command. Use 'q' to quit, 'r' for random, or enter a number."
            )


def dataset_statistics(dataset_path):
    """Show dataset statistics"""
    dataset_path = Path(dataset_path)

    for split in ["train", "valid"]:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"

        if not images_dir.exists():
            continue

        # Count images and labels
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []

        annotated_count = 0
        bbox_sizes = []

        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                annotation = load_yolo_annotation(label_file)
                if annotation:
                    annotated_count += 1
                    bbox_sizes.append(annotation["width"] * annotation["height"])

        print(f"\n=== {split.upper()} SET ===")
        print(f"Images: {len(image_files)}")
        print(f"Labels: {len(label_files)}")
        print(f"Annotated: {annotated_count}")
        if bbox_sizes:
            print(f"Avg bbox size: {np.mean(bbox_sizes):.4f} (normalized)")
            print(f"Min bbox size: {np.min(bbox_sizes):.4f}")
            print(f"Max bbox size: {np.max(bbox_sizes):.4f}")


def main():
    """Main function with examples"""
    DATASET_PATH = "./dataset"  # Update this path

    print("=== Dataset Viewer ===")

    # Show dataset statistics
    print("Dataset Statistics:")
    dataset_statistics(DATASET_PATH)

    # View random samples
    print("\nShowing random training samples...")
    view_random_samples(DATASET_PATH, split="valid", num_samples=6)

    # Interactive browsing
    print("\nStarting interactive browser...")
    browse_dataset(DATASET_PATH, split="valid")


if __name__ == "__main__":
    main()
