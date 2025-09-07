from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks

# Configuration
FILE_PATH = "1.mp4"  # INPUT VIDEO PATH
OUTPUT_PATH = "output/test.mp4"
MODEL_PATH = "model/barbell_detector_model_v1.h5"  # TensorFlow model path

TRACKING_ALGORITHMS = ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "MOSSE", "CSRT"]
DEFAULT_TRACKER = "CSRT"  # CSRT is generally best for accuracy, KCF for speed

CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection


class BarbellTracker:
    def __init__(
        self,
        video_path,
        output_path,
        model_path,
        tracker_type="CSRT",
        input_size=(312, 312),
    ):
        """Initialize the TensorFlow-based barbell tracker"""
        self.video_path = video_path
        self.output_path = output_path
        self.model_path = model_path
        self.tracker_type = tracker_type
        self.input_size = input_size
        self.confidence_threshold = CONFIDENCE_THRESHOLD

        self.tracker = None
        self.model = None
        self.center_points = []
        self.frame_count = 0
        self.detection_interval = 30  # Re-detect every N frames for correction
        self.detections_log = []  # Store all detections for analysis

    def load_tensorflow_model(self):
        """Load the TensorFlow model for barbell detection"""
        try:
            print(f"Loading TensorFlow model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully!")

            # Print model summary for debugging
            print("\nModel Input Shape:", self.model.input_shape)
            print(
                "Model Output Shape:", [output.shape for output in self.model.outputs]
            )
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def preprocess_frame_for_model(self, frame):
        """Preprocess frame for TensorFlow model input"""
        # Store original shape for later conversion
        original_shape = frame.shape[:2]  # (height, width)

        # Resize for model input
        frame_resized = cv2.resize(
            frame, self.input_size, interpolation=cv2.INTER_LINEAR
        )

        # Normalize to [0, 1]
        frame_normalized = frame_resized.astype(np.float32) / 255.0

        # Add batch dimension
        frame_batch = np.expand_dims(frame_normalized, axis=0)

        return frame_batch, original_shape

    def detect_barbell_tensorflow(self, frame):
        """Detect barbell in frame using TensorFlow model"""
        # Preprocess frame
        frame_batch, original_shape = self.preprocess_frame_for_model(frame)

        # Run inference
        try:
            bbox_pred, confidence_pred = self.model.predict(frame_batch, verbose=0)

            # Extract predictions
            bbox = bbox_pred[0]  # [x_center, y_center, width, height] normalized
            confidence = float(confidence_pred[0][0])

            if confidence > self.confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                height, width = original_shape

                x_center = bbox[0] * width
                y_center = bbox[1] * height
                box_width = bbox[2] * width
                box_height = bbox[3] * height

                # Calculate corner coordinates for tracker (x, y, width, height)
                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)

                # Ensure coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                box_width = min(box_width, width - x1)
                box_height = min(box_height, height - y1)

                tracker_bbox = (x1, y1, int(box_width), int(box_height))

                print(f"Barbell detected with confidence: {confidence:.3f}")
                print(
                    f"  Position: ({x1}, {y1}), Size: {int(box_width)}x{int(box_height)}"
                )

                # Log detection
                self.detections_log.append(
                    {
                        "frame": self.frame_count,
                        "confidence": confidence,
                        "bbox": tracker_bbox,
                        "center": (int(x_center), int(y_center)),
                    }
                )

                return tracker_bbox, confidence
            else:
                print(f"Detection confidence too low: {confidence:.3f}")
                return None, confidence

        except Exception as e:
            print(f"Error during detection: {e}")
            return None, 0

    def initiate_tracker(self):
        """Initialize the selected tracking algorithm"""
        trackers = {
            # "BOOSTING": cv2.TrackerBoosting_create,
            # "MIL": cv2.TrackerMIL_create,
            # "KCF": cv2.TrackerKCF_create,
            # "TLD": cv2.TrackerTLD_create,
            # "MEDIANFLOW": cv2.TrackerMedianFlow_create,
            # "MOSSE": cv2.TrackerMOSSE_create,
            "CSRT": cv2.TrackerCSRT_create,
        }

        if self.tracker_type in trackers:
            self.tracker = trackers[self.tracker_type]()
            print(f"Using {self.tracker_type} tracker")
        else:
            print(f"Unknown tracker type: {self.tracker_type}. Using CSRT as default.")
            self.tracker = cv2.TrackerCSRT_create()

    def create_video_writer(self, cap):
        """Create video writer with same properties as input video"""
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nVideo Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")

        # Ensure output directory exists
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        if writer.isOpened():
            print(f"Video writer created successfully")
        else:
            print("Failed to create video writer")

        return writer, total_frames

    def draw_tracking_info(self, frame, bbox, confidence=None, status="TRACKING"):
        """Draw bounding box, trajectory, and info on frame"""
        if bbox is not None:
            x, y, w, h = [int(i) for i in bbox]

            # Choose color based on status
            if status == "DETECTED":
                color = (0, 255, 0)  # Green for fresh detection
            elif status == "RE-DETECTED":
                color = (0, 255, 255)  # Yellow for re-detection
            else:
                color = (255, 0, 0)  # Blue for tracking

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Calculate and draw center point
            center = (int(x + w / 2), int(y + h / 2))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Add to trajectory
            self.center_points.append(center)

            # Draw trajectory (limit to last 150 points for performance)
            points_to_draw = (
                self.center_points[-150:]
                if len(self.center_points) > 150
                else self.center_points
            )

            # Draw with gradient effect (older points are dimmer)
            for i in range(1, len(points_to_draw)):
                # Calculate color intensity based on position in trajectory
                intensity = int(255 * (i / len(points_to_draw)))
                color_trajectory = (intensity, 0, 255 - intensity)
                thickness = max(1, int(3 * (i / len(points_to_draw))))
                cv2.line(
                    frame,
                    points_to_draw[i - 1],
                    points_to_draw[i],
                    color_trajectory,
                    thickness,
                )

            # Add text info
            info_text = f"Frame: {self.frame_count} | Status: {status}"
            if confidence is not None:
                info_text += f" | Conf: {confidence:.3f}"

            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(
                info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (10, 10), (20 + text_width, 40), (0, 0, 0), -1)
            cv2.putText(
                frame,
                info_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Add tracker type
            cv2.putText(
                frame,
                f"Tracker: {self.tracker_type}",
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Add barbell position info
            pos_text = f"Position: ({center[0]}, {center[1]})"
            cv2.putText(
                frame,
                pos_text,
                (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        else:
            # No detection/tracking
            cv2.putText(
                frame,
                f"Frame: {self.frame_count} | NO DETECTION",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        return frame

    def process_video(self):
        """Main processing loop for video tracking"""
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {self.video_path}")
            return False

        # Load TensorFlow model
        if not self.load_tensorflow_model():
            return False

        # Create video writer
        writer, total_frames = self.create_video_writer(cap)

        # Get first frame for initial detection
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Cannot read first frame")
            return False

        # Detect barbell in first frame using TensorFlow
        print("\n=== Initial Detection ===")
        initial_bbox, initial_conf = self.detect_barbell_tensorflow(first_frame)

        if initial_bbox is None:
            print("\nNo barbell detected in first frame. Trying manual selection...")
            print("Please select the barbell region in the window...")
            initial_bbox = cv2.selectROI("Select Barbell", first_frame, False)
            cv2.destroyWindow("Select Barbell")

            if initial_bbox == (0, 0, 0, 0):
                print("No region selected. Exiting.")
                return False
            initial_conf = 0.0
            status = "MANUAL"
        else:
            status = "DETECTED"

        # Initialize tracker
        self.initiate_tracker()
        self.tracker.init(first_frame, initial_bbox)

        # Process first frame
        first_frame = self.draw_tracking_info(
            first_frame, initial_bbox, initial_conf, status
        )
        writer.write(first_frame)
        cv2.imshow("Barbell Tracking", first_frame)

        # Process remaining frames
        print("\n=== Processing Video ===")
        print("Press 'q' to quit, 'd' to force re-detection")

        tracking_failures = 0
        last_good_bbox = initial_bbox

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            # Progress indicator
            if self.frame_count % 30 == 0:
                progress = (self.frame_count / total_frames) * 100
                print(
                    f"Progress: {self.frame_count}/{total_frames} frames ({progress:.1f}%)"
                )

            # Update tracker
            success, bbox = self.tracker.update(frame)

            # Check for forced re-detection or periodic re-detection
            force_redetect = self.frame_count % self.detection_interval == 0

            if force_redetect or not success:
                if not success:
                    tracking_failures += 1
                    print(
                        f"Tracking lost at frame {self.frame_count} (failure #{tracking_failures})"
                    )

                # Try to re-detect using TensorFlow
                detected_bbox, detected_conf = self.detect_barbell_tensorflow(frame)

                if (
                    detected_bbox is not None
                    and detected_conf > self.confidence_threshold
                ):
                    # Re-initialize tracker with new detection
                    self.tracker = None
                    self.initiate_tracker()
                    self.tracker.init(frame, detected_bbox)
                    bbox = detected_bbox
                    last_good_bbox = detected_bbox
                    success = True
                    status = "RE-DETECTED"
                    print(f"  Re-detected with confidence {detected_conf:.3f}")
                    tracking_failures = 0  # Reset failure counter
                else:
                    # Use last known good position
                    bbox = last_good_bbox
                    status = "LOST"
                    print(f"  Re-detection failed, using last known position")
            else:
                last_good_bbox = bbox
                status = "TRACKING"

            # Draw visualization
            frame = self.draw_tracking_info(
                frame, bbox if success else None, confidence=None, status=status
            )

            # Write frame
            writer.write(frame)

            # Display frame
            cv2.imshow("Barbell Tracking", frame)

            # Check for user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nProcessing interrupted by user")
                break
            elif key == ord("d"):
                print("\nForced re-detection requested")
                detected_bbox, detected_conf = self.detect_barbell_tensorflow(frame)
                if detected_bbox is not None:
                    self.tracker = None
                    self.initiate_tracker()
                    self.tracker.init(frame, detected_bbox)

        # Cleanup
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        print(f"\n=== Processing Complete ===")
        print(f"Output saved to: {self.output_path}")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total detections: {len(self.detections_log)}")
        print(f"Trajectory points recorded: {len(self.center_points)}")
        print(f"Tracking failures: {tracking_failures}")

        return True

    def analyze_trajectory(self):
        """Analyze the tracked trajectory for squat metrics"""
        if len(self.center_points) < 10:
            print("Insufficient data for trajectory analysis")
            return None

        points = np.array(self.center_points)

        # Extract y-coordinates (vertical movement)
        y_coords = points[:, 1]

        # Smooth the trajectory
        from scipy.ndimage import gaussian_filter1d

        y_smooth = gaussian_filter1d(y_coords, sigma=2)

        # Calculate metrics
        y_range = np.max(y_smooth) - np.min(y_smooth)

        # Detect peaks and valleys for rep counting
        try:
            # Find valleys (bottom of squat)
            valleys, valley_props = find_peaks(
                -y_smooth, distance=20, prominence=y_range * 0.15
            )
            # Find peaks (top of squat)
            peaks, peak_props = find_peaks(
                y_smooth, distance=20, prominence=y_range * 0.15
            )

            # Count complete reps (a valley followed by a peak)
            num_reps = min(len(valleys), len(peaks))

            # Calculate average rep duration
            if num_reps > 1:
                avg_rep_frames = self.frame_count / num_reps
            else:
                avg_rep_frames = 0

            # Calculate velocity metrics
            velocities = np.diff(y_smooth)
            avg_velocity = np.mean(np.abs(velocities))
            max_velocity = np.max(np.abs(velocities))

            print("\n=== Trajectory Analysis ===")
            print(f"Vertical range of motion: {y_range:.1f} pixels")
            print(f"Number of complete reps: {num_reps}")
            print(f"Average rep duration: {avg_rep_frames:.1f} frames")
            print(f"Average velocity: {avg_velocity:.2f} pixels/frame")
            print(f"Max velocity: {max_velocity:.2f} pixels/frame")

            # Confidence statistics from detections
            if self.detections_log:
                confidences = [d["confidence"] for d in self.detections_log]
                print(f"\n=== Detection Statistics ===")
                print(f"Total detections: {len(self.detections_log)}")
                print(f"Average confidence: {np.mean(confidences):.3f}")
                print(f"Min confidence: {np.min(confidences):.3f}")
                print(f"Max confidence: {np.max(confidences):.3f}")

            return {
                "vertical_range": y_range,
                "num_reps": num_reps,
                "avg_rep_frames": avg_rep_frames,
                "trajectory_points": points,
                "valleys": valleys,
                "peaks": peaks,
                "avg_velocity": avg_velocity,
                "max_velocity": max_velocity,
            }

        except Exception as e:
            print(f"Error in trajectory analysis: {e}")
            return None

    def plot_trajectory_analysis(self, save_path=None):
        """Create visualization plots for trajectory analysis"""
        if len(self.center_points) < 10:
            print("Insufficient data for plotting")
            return

        points = np.array(self.center_points)
        y_coords = points[:, 1]

        # Smooth trajectory
        from scipy.ndimage import gaussian_filter1d

        y_smooth = gaussian_filter1d(y_coords, sigma=2)

        # Detect peaks and valleys
        y_range = np.max(y_smooth) - np.min(y_smooth)
        valleys, _ = find_peaks(-y_smooth, distance=20, prominence=y_range * 0.15)
        peaks, _ = find_peaks(y_smooth, distance=20, prominence=y_range * 0.15)

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Raw vertical position over time
        axes[0, 0].plot(y_coords, "b-", alpha=0.5, label="Raw")
        axes[0, 0].plot(y_smooth, "r-", linewidth=2, label="Smoothed")
        axes[0, 0].plot(
            valleys, y_smooth[valleys], "gv", markersize=10, label="Valleys (Bottom)"
        )
        axes[0, 0].plot(
            peaks, y_smooth[peaks], "r^", markersize=10, label="Peaks (Top)"
        )
        axes[0, 0].set_xlabel("Frame")
        axes[0, 0].set_ylabel("Vertical Position (pixels)")
        axes[0, 0].set_title("Barbell Vertical Movement")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].invert_yaxis()  # Invert because y increases downward in images

        # Plot 2: 2D trajectory
        x_coords = points[:, 0]
        axes[0, 1].plot(x_coords, y_coords, "b-", alpha=0.3)
        axes[0, 1].scatter(
            x_coords[::10],
            y_coords[::10],
            c=range(0, len(x_coords), 10),
            cmap="viridis",
            s=20,
        )
        axes[0, 1].set_xlabel("Horizontal Position (pixels)")
        axes[0, 1].set_ylabel("Vertical Position (pixels)")
        axes[0, 1].set_title("2D Barbell Trajectory")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_aspect("equal")

        # Plot 3: Velocity over time
        velocity = np.diff(y_smooth)
        axes[1, 0].plot(velocity, "g-")
        axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
        axes[1, 0].set_xlabel("Frame")
        axes[1, 0].set_ylabel("Vertical Velocity (pixels/frame)")
        axes[1, 0].set_title("Barbell Velocity")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Detection confidence over time
        if self.detections_log:
            det_frames = [d["frame"] for d in self.detections_log]
            det_confs = [d["confidence"] for d in self.detections_log]
            axes[1, 1].scatter(det_frames, det_confs, c="blue", alpha=0.6)
            axes[1, 1].axhline(
                y=self.confidence_threshold,
                color="r",
                linestyle="--",
                label=f"Threshold ({self.confidence_threshold})",
            )
            axes[1, 1].set_xlabel("Frame")
            axes[1, 1].set_ylabel("Detection Confidence")
            axes[1, 1].set_title("Model Detection Confidence")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])

        plt.suptitle("Barbell Tracking Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Analysis plot saved to: {save_path}")

        plt.show()


def main():
    """Main function to run the TensorFlow barbell tracker"""

    # Configuration
    tracker = BarbellTracker(
        video_path=FILE_PATH,
        output_path=OUTPUT_PATH,
        model_path=MODEL_PATH,
        tracker_type=DEFAULT_TRACKER,
        input_size=(312, 312),  # Match your model's input size
    )

    # Process video
    if tracker.process_video():
        # Analyze trajectory
        try:
            analysis_results = tracker.analyze_trajectory()

            # Create visualization plots
            tracker.plot_trajectory_analysis(save_path="output/trajectory_analysis.png")

        except ImportError as e:
            print(f"\nNote: Install scipy for advanced analysis: pip install scipy")
            print(f"Error: {e}")
    else:
        print("Video processing failed")


if __name__ == "__main__":
    main()
