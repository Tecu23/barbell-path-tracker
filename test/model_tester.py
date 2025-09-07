from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class BarbellModelTester:
    def __init__(self, model_path, input_size=(312, 312)):
        """Initialize the model tester"""
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = 0.5

        # Load the trained model
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")

    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = original_img.shape[:2]  # (height, width)

        # Resize for model input
        img_resized = cv2.resize(
            original_img, self.input_size, interpolation=cv2.INTER_LINEAR
        )
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        return img_batch, original_img, original_shape

    def predict_single_image(self, image_path):
        """Predict barbell location in a single image"""
        # Preprocess
        img_batch, original_img, original_shape = self.preprocess_image(image_path)

        # Predict
        bbox_pred, confidence_pred = self.model.predict(img_batch, verbose=0)

        # Extract predictions
        bbox = bbox_pred[0]  # [x_center, y_center, width, height] normalized
        confidence = float(confidence_pred[0][0])

        # Convert to pixel coordinates
        height, width = original_shape
        if confidence > self.confidence_threshold:
            x_center = bbox[0] * width
            y_center = bbox[1] * height
            box_width = bbox[2] * width
            box_height = bbox[3] * height

            # Calculate corner coordinates
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            pixel_bbox = (x1, y1, x2, y2)
        else:
            pixel_bbox = None

        return {
            "confidence": confidence,
            "bbox_normalized": bbox,
            "bbox_pixels": pixel_bbox,
            "original_image": original_img,
            "image_path": image_path,
        }

    def visualize_prediction(self, result, save_path=None):
        """Visualize prediction on image"""
        img = result["original_image"].copy()
        confidence = result["confidence"]
        bbox_pixels = result["bbox_pixels"]

        plt.figure(figsize=(10, 8))
        plt.imshow(img)

        # Draw bounding box if detected
        if bbox_pixels is not None:
            x1, y1, x2, y2 = bbox_pixels
            width = x2 - x1
            height = y2 - y1

            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height, linewidth=3, edgecolor="red", facecolor="none"
            )
            plt.gca().add_patch(rect)

            # Add center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            plt.plot(center_x, center_y, "go", markersize=8)

            # Add confidence text
            plt.text(
                x1,
                y1 - 10,
                f"Barbell: {confidence:.3f}",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                fontsize=12,
                color="white",
                weight="bold",
            )
        else:
            plt.text(
                10,
                30,
                f"No barbell detected (conf: {confidence:.3f})",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                fontsize=12,
                color="white",
                weight="bold",
            )

        plt.title(f'Barbell Detection - {Path(result["image_path"]).name}')
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to: {save_path}")

        plt.show()

        return result

    def test_single_image(self, image_path, visualize=True, save_path=None):
        """Test model on a single image"""
        print(f"\n=== Testing: {Path(image_path).name} ===")

        # Predict
        result = self.predict_single_image(image_path)

        # Print results
        print(f"Confidence: {result['confidence']:.4f}")
        if result["bbox_pixels"]:
            x1, y1, x2, y2 = result["bbox_pixels"]
            print(f"Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"Center: ({(x1+x2)/2:.1f}, {(y1+y2)/2:.1f})")
            print(f"Size: {x2-x1} x {y2-y1} pixels")
        else:
            print("No barbell detected")

        # Visualize if requested
        if visualize:
            self.visualize_prediction(result, save_path)

        return result

    def load_yolo_annotation(self, label_path):
        """Load YOLO format annotation and convert to pixel coordinates"""
        if not label_path.exists():
            return None

        with open(label_path, "r") as f:
            line = f.readline().strip()

        if not line:
            return None

        parts = line.split()
        if len(parts) < 5:
            return None

        # YOLO format: class_id x_center y_center width height (normalized)
        return {
            "class_id": int(parts[0]),
            "x_center": float(parts[1]),
            "y_center": float(parts[2]),
            "width": float(parts[3]),
            "height": float(parts[4]),
        }

    def yolo_to_pixel_bbox(self, yolo_annotation, image_shape):
        """Convert YOLO normalized coordinates to pixel bounding box"""
        if yolo_annotation is None:
            return None

        height, width = image_shape

        x_center = yolo_annotation["x_center"] * width
        y_center = yolo_annotation["y_center"] * height
        box_width = yolo_annotation["width"] * width
        box_height = yolo_annotation["height"] * height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        return (x1, y1, x2, y2)

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # bbox format: (x1, y1, x2, y2)
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        # Calculate intersection area
        if x2_i <= x1_i or y2_i <= y1_i:
            intersection = 0
        else:
            intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        # Calculate IoU
        if union == 0:
            return 0
        return intersection / union

    def test_dataset(self, test_dir, max_images=10):
        """Test model on multiple images from a directory"""
        test_dir = Path(test_dir)

        # Find all image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(test_dir.glob(f"*{ext}"))
            image_files.extend(test_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"No images found in {test_dir}")
            return

        print(
            f"\n=== Testing {min(len(image_files), max_images)} \
            images from {test_dir} ==="
        )

        results = []
        detections = 0

        for i, img_path in enumerate(image_files[:max_images]):
            result = self.predict_single_image(img_path)
            results.append(result)

            if result["confidence"] > self.confidence_threshold:
                detections += 1

            print(
                f"{i+1:2d}. {img_path.name:30s} | Conf: {result['confidence']:.3f} | "
                f"{'✓ DETECTED' if result['bbox_pixels'] else '✗ No detection'}"
            )

        print(f"\nSummary: {detections}/{len(results)} images had barbell detections")
        print(f"Detection rate: {detections/len(results)*100:.1f}%")

        return results

    def test_with_ground_truth(self, images_dir, labels_dir, max_images=20):
        """Test model against ground truth annotations (YOLO format)"""
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        print(f"\n=== Testing with Ground Truth ===")

        # Find image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        results = []
        correct_detections = 0
        false_positives = 0
        false_negatives = 0

        for i, img_path in enumerate(image_files[:max_images]):
            # Load ground truth
            label_path = labels_dir / f"{img_path.stem}.txt"
            has_ground_truth = False
            gt_bbox = None

            if label_path.exists():
                with open(label_path, "r") as f:
                    line = f.readline().strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        # YOLO format: class_id x_center y_center width height
                        gt_bbox = [float(x) for x in parts[1:5]]
                        has_ground_truth = True

            # Get prediction
            result = self.predict_single_image(img_path)
            has_prediction = result["confidence"] > self.confidence_threshold

            # Calculate metrics
            if has_ground_truth and has_prediction:
                correct_detections += 1
                status = "✓ TRUE POSITIVE"
            elif has_ground_truth and not has_prediction:
                false_negatives += 1
                status = "✗ FALSE NEGATIVE"
            elif not has_ground_truth and has_prediction:
                false_positives += 1
                status = "✗ FALSE POSITIVE"
            else:
                status = "✓ TRUE NEGATIVE"

            print(
                f"{i+1:2d}. {img_path.name:30s}\
                | Conf: {result['confidence']:.3f} | {status}"
            )

            results.append(
                {
                    "image_path": img_path,
                    "prediction": result,
                    "ground_truth": gt_bbox,
                    "has_ground_truth": has_ground_truth,
                    "has_prediction": has_prediction,
                }
            )

        # Calculate metrics
        total = len(results)
        true_positives = correct_detections
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print("\n=== METRICS ===")
        print(f"Total images: {total}")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1_score:.3f}")

        return results

    def test_on_training_set(
        self, dataset_path, split="train", max_images=50, iou_threshold=0.5
    ):
        """Test model on training dataset with detailed metrics"""
        dataset_path = Path(dataset_path)
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"

        print(f"\n=== Testing on {split.upper()} Dataset ===")

        if not images_dir.exists():
            print(f"Images directory not found: {images_dir}")
            return None

        if not labels_dir.exists():
            print(f"Labels directory not found: {labels_dir}")
            return None

        # Find all image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            print("No images found!")
            return None

        # Limit number of images
        image_files = image_files[:max_images]
        print(f"Testing on {len(image_files)} images...")

        # Metrics storage
        results = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        ious = []
        confidences = []

        for i, img_path in enumerate(image_files):
            # Load ground truth
            label_path = labels_dir / f"{img_path.stem}.txt"
            gt_annotation = self.load_yolo_annotation(label_path)

            # Get model prediction
            prediction = self.predict_single_image(img_path)

            # Load original image to get dimensions
            original_img = prediction["original_image"]
            image_shape = original_img.shape[:2]

            # Convert ground truth to pixel coordinates
            gt_bbox_pixels = self.yolo_to_pixel_bbox(gt_annotation, image_shape)
            print(gt_annotation, prediction["bbox_normalized"])
            pred_annotation = {
                "class_id": 0,
                "x_center": prediction["bbox_normalized"][0],
                "y_center": prediction["bbox_normalized"][1],
                "width": prediction["bbox_normalized"][2],
                "height": prediction["bbox_normalized"][3],
            }

            pred_bbox_pixels = self.yolo_to_pixel_bbox(pred_annotation, image_shape)

            # Determine ground truth and prediction status
            has_gt = gt_bbox_pixels is not None
            has_pred = (
                pred_bbox_pixels is not None
                and prediction["confidence"] > self.confidence_threshold
            )

            # Calculate IoU if both exist
            iou = 0.0
            if has_gt and has_pred:
                iou = self.calculate_iou(gt_bbox_pixels, pred_bbox_pixels)
                ious.append(iou)

            # Classification based on IoU threshold
            if has_gt and has_pred and iou >= iou_threshold:
                classification = "TP"  # True Positive
                true_positives += 1
            elif has_gt and (not has_pred or iou < iou_threshold):
                classification = "FN"  # False Negative
                false_negatives += 1
            elif not has_gt and has_pred:
                classification = "FP"  # False Positive
                false_positives += 1
            else:
                classification = "TN"  # True Negative
                true_negatives += 1

            confidences.append(prediction["confidence"])

            # Store detailed results
            result = {
                "image_path": img_path,
                "ground_truth": gt_annotation,
                "gt_bbox_pixels": gt_bbox_pixels,
                "prediction": prediction,
                "pred_bbox_pixels": pred_bbox_pixels,
                "iou": iou,
                "classification": classification,
                "confidence": prediction["confidence"],
            }
            results.append(result)

            # Progress and detailed output
            status_icon = {"TP": "✓", "FN": "✗", "FP": "⚠", "TN": "○"}[classification]
            print(
                f"{i+1:3d}. {img_path.name:35s} | Conf: {prediction['confidence']:.3f} | "
                f"IoU: {iou:.3f} | {status_icon} {classification}"
            )

        # Calculate comprehensive metrics
        total = len(results)
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0

        avg_iou = np.mean(ious) if ious else 0
        avg_confidence = np.mean(confidences)
        detection_rate = (true_positives + false_positives) / total * 100

        print(f"\n=== DETAILED METRICS ===")
        print(f"Total images: {total}")
        print(f"True Positives (TP): {true_positives}")
        print(f"False Positives (FP): {false_positives}")
        print(f"False Negatives (FN): {false_negatives}")
        print(f"True Negatives (TN): {true_negatives}")
        print(f"")
        print(f"Precision: {precision:.4f} (TP / (TP + FP))")
        print(f"Recall: {recall:.4f} (TP / (TP + FN))")
        print(f"F1-Score: {f1_score:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"")
        print(f"Average IoU: {avg_iou:.4f} (for detected objects)")
        print(f"Average Confidence: {avg_confidence:.4f}")
        print(f"Detection Rate: {detection_rate:.1f}%")
        print(f"IoU Threshold: {iou_threshold}")

        return {
            "results": results,
            "metrics": {
                "total": total,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_negatives": true_negatives,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "accuracy": accuracy,
                "avg_iou": avg_iou,
                "avg_confidence": avg_confidence,
                "detection_rate": detection_rate,
                "iou_threshold": iou_threshold,
            },
        }

    def create_error_analysis_grid(
        self, test_results, error_type="FN", max_images=9, save_path=None
    ):
        """Create a grid showing specific types of errors"""
        # Filter results by error type
        filtered_results = [
            r for r in test_results["results"] if r["classification"] == error_type
        ]

        if not filtered_results:
            print(f"No {error_type} cases found!")
            return

        # Limit to max_images
        filtered_results = filtered_results[:max_images]

        # Calculate grid size
        n_images = len(filtered_results)
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))

        # Create subplot grid
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        error_names = {
            "TP": "True Positives (Correct)",
            "FP": "False Positives (Over-detection)",
            "FN": "False Negatives (Missed)",
            "TN": "True Negatives (Correct No-detection)",
        }

        for i in range(rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            if i < len(filtered_results):
                result = filtered_results[i]
                img = result["prediction"]["original_image"]

                # Show image
                ax.imshow(img)

                # Draw ground truth (green) if exists
                if result["gt_bbox_pixels"]:
                    x1, y1, x2, y2 = result["gt_bbox_pixels"]
                    width, height = x2 - x1, y2 - y1
                    rect = patches.Rectangle(
                        (x1, y1),
                        width,
                        height,
                        linewidth=2,
                        edgecolor="green",
                        facecolor="none",
                        linestyle="--",
                    )
                    ax.add_patch(rect)

                # Draw prediction (red) if exists
                if result["pred_bbox_pixels"]:
                    x1, y1, x2, y2 = result["pred_bbox_pixels"]
                    width, height = x2 - x1, y2 - y1
                    rect = patches.Rectangle(
                        (x1, y1),
                        width,
                        height,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                    )
                    ax.add_patch(rect)

                # Title with details
                confidence = result["confidence"]
                iou = result["iou"]
                filename = Path(result["image_path"]).name
                ax.set_title(
                    f"{filename}\nConf: {confidence:.3f} | IoU: {iou:.3f}",
                    fontsize=9,
                    weight="bold",
                )
            else:
                ax.axis("off")

            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(
            f"{error_names[error_type]} - {len(filtered_results)} cases",
            fontsize=16,
            weight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Error analysis grid saved to: {save_path}")

        plt.show()

    def create_detection_grid(
        self, test_dir, max_images=9, save_path="test/output/detection_grid.png"
    ):
        """Create a grid visualization of multiple detections"""
        test_dir = Path(test_dir)

        # Find image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(test_dir.glob(f"*{ext}"))

        if len(image_files) == 0:
            print("No images found!")
            return

        # Limit to max_images
        image_files = image_files[:max_images]

        # Calculate grid size
        n_images = len(image_files)
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))

        # Create subplot grid
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            if i < len(image_files):
                # Get prediction
                result = self.predict_single_image(image_files[i])
                img = result["original_image"]
                confidence = result["confidence"]
                bbox_pixels = result["bbox_pixels"]

                # Show image
                ax.imshow(img)

                # Draw bounding box if detected
                if bbox_pixels is not None:
                    x1, y1, x2, y2 = bbox_pixels
                    width = x2 - x1
                    height = y2 - y1

                    rect = patches.Rectangle(
                        (x1, y1),
                        width,
                        height,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                    )
                    ax.add_patch(rect)

                    # Add center point
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    ax.plot(center_x, center_y, "go", markersize=6)

                # Title with confidence
                color = "green" if confidence > self.confidence_threshold else "red"
                ax.set_title(
                    f"{Path(image_files[i]).name}\nConf: {confidence:.3f}",
                    fontsize=10,
                    color=color,
                    weight="bold",
                )
            else:
                ax.axis("off")

            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Detection grid saved to: {save_path}")


def main():
    """Example usage of the BarbellModelTester"""

    # Initialize tester with your trained model
    MODEL_PATH = "model/barbell_detector_model_v1.h5"  # or "barbell_detector_final.h5"
    DATASET_PATH = "dataset"

    try:
        tester = BarbellModelTester(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model file exists and training completed successfully.")
        return

    # Test options - uncomment what you want to test:

    # 1. Test on training set with detailed metrics
    print("=== TESTING ON TRAINING SET ===")
    train_results = tester.test_on_training_set(
        DATASET_PATH, split="train", max_images=50, iou_threshold=0.5
    )

    # 2. Test on validation set
    print("\n=== TESTING ON VALIDATION SET ===")
    val_results = tester.test_on_training_set(
        DATASET_PATH, split="valid", max_images=30, iou_threshold=0.5
    )

    # 3. Visualize prediction vs ground truth for specific images
    # tester.visualize_prediction_vs_groundtruth(
    #     "../dataset/train/images/image_001.jpg",
    #     Path("../dataset/train/labels")
    # )

    # 4. Create error analysis grids
    if train_results:
        print("\n=== ERROR ANALYSIS ===")
        # Show false negatives (missed detections)
        tester.create_error_analysis_grid(
            train_results,
            error_type="FN",
            max_images=6,
            save_path="test/output/false_negatives.png",
        )

        # Show false positives (over-detections)
        tester.create_error_analysis_grid(
            train_results,
            error_type="FP",
            max_images=6,
            save_path="test/output/false_positives.png",
        )

        # Show true positives (correct detections)
        tester.create_error_analysis_grid(
            train_results,
            error_type="TP",
            max_images=6,
            save_path="test/output/true_positives.png",
        )

    # 5. Create detection grid visualization
    print("\nCreating detection grid...")
    tester.create_detection_grid(
        f"{DATASET_PATH}/valid/images",
        max_images=9,
        save_path="test/output/validation_results.png",
    )


if __name__ == "__main__":
    main()
