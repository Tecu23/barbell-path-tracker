import cv2
import numpy as np


class DataAugmentation:
    def __init__(self, augment_prob=0.8):
        self.augment_prob = augment_prob

    def augment_batch(self, images, bboxes, confidences):
        """Apply augmentations to a batch"""
        augmented_images = []
        augmented_bboxes = []
        augmented_confidences = []

        for img, bbox, conf in zip(images, bboxes, confidences):
            if np.random.random() < self.augment_prob and conf > 0:
                img_aug, bbox_aug = self.augment_single(img, bbox)
                augmented_images.append(img_aug)
                augmented_bboxes.append(bbox_aug)
                augmented_confidences.append(conf)
            else:
                augmented_images.append(img)
                augmented_bboxes.append(bbox)
                augmented_confidences.append(conf)

        return (
            np.array(augmented_images),
            np.array(augmented_bboxes),
            np.array(augmented_confidences),
        )

    def augment_single(self, image, bbox):
        """Apply random augmentations to single image and bbox"""
        img = image.copy()
        x_center, y_center, width, height = bbox.copy()

        # Random brightness (50% chance)
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.7, 1.3)
            img = np.clip(img * brightness, 0, 1)

        # Random contrast (30% chance)
        if np.random.random() < 0.3:
            contrast = np.random.uniform(0.8, 1.2)
            img = np.clip((img - 0.5) * contrast + 0.5, 0, 1)

        # Random horizontal flip (30% chance)
        if np.random.random() < 0.3:
            img = np.fliplr(img)
            x_center = 1 - x_center  # Flip x coordinate

        # Random noise (20% chance)
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 0.02, img.shape)
            img = np.clip(img + noise, 0, 1)

        # Random rotation (small angles, 20% chance)
        if np.random.random() < 0.2:
            angle = np.random.uniform(-5, 5)
            center = (img.shape[1] // 2, img.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

        return img, [x_center, y_center, width, height]
