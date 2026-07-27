#!/usr/bin/env python3
"""
Preprocessing pipeline for museum specimen label photographs.

Handles deskewing, denoising, contrast enhancement, and optional cropping
for OCR preparation. Supports both manual bounding boxes and auto-detection.
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    target_size: int = 1024
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    denoise_strength: int = 10
    deskew_angle_threshold: float = 0.5  # Minimum angle to apply deskew
    min_contour_area_ratio: float = 0.01  # Min area ratio for contour detection


class PreprocessingPipeline:
    """Main preprocessing pipeline for OCR images."""

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize preprocessing pipeline.

        Args:
            config: Preprocessing configuration parameters.
        """
        self.config = config
        self.stats = {
            "processed": 0,
            "failed": 0,
            "total_time": 0.0,
            "deskewed": 0,
            "cropped": 0,
        }

    def load_image(self, image_path: Path) -> np.ndarray:
        """
        Load image with OpenCV.

        Args:
            image_path: Path to image file.

        Returns:
            RGB image as numpy array.

        Raises:
            ValueError: If image cannot be loaded.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def auto_detect_label_boundary(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Auto-detect the largest rectangular contour as label boundary.

        Args:
            image: RGB image array.

        Returns:
            Binary mask of detected label region, or None if not found.

        Assumption:
            - Label is the largest roughly-rectangular object
            - Background is relatively uniform
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Adaptive threshold to handle varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations to connect text regions
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest_contour) / (image.shape[0] * image.shape[1])

        # Filter by minimum area ratio
        if area_ratio < self.config.min_contour_area_ratio:
            logger.debug(f"Largest contour too small: {area_ratio:.3f}")
            return None

        # Create mask from largest contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        return mask

    def deskew_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Deskew image using minAreaRect on thresholded text contours.

        Args:
            image: RGB image array.

        Returns:
            Tuple of (deskewed image, rotation angle in degrees).

        Assumption:
            - Text is the main content and provides good edge detection
            - Image may be rotated up to ±45 degrees
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Threshold to find text regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.debug("No contours found for deskewing")
            return image, 0.0

        # Filter out small contours
        min_contour_area = 100
        filtered_contours = [
            cnt for cnt in contours
            if cv2.contourArea(cnt) > min_contour_area
        ]

        if not filtered_contours:
            logger.debug("No significant contours found for deskewing")
            return image, 0.0

        # Find rotated rectangle for each contour and get angles
        angles = []
        for contour in filtered_contours[:20]:  # Limit to top 20 for performance
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            angles.append(angle)

        # Use median angle to be robust to outliers
        median_angle = np.median(angles)

        # Convert angle to rotation angle
        if median_angle < -45:
            median_angle = 90 + median_angle
        elif median_angle > 45:
            median_angle = median_angle - 90

        # Skip if angle is below threshold
        if abs(median_angle) < self.config.deskew_angle_threshold:
            return image, 0.0

        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        return rotated, median_angle

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE and denoising for contrast enhancement.

        Args:
            image: RGB image array.

        Returns:
            Enhanced image array.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )
        l_enhanced = clahe.apply(l)

        # Merge back
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        # Apply denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            enhanced,
            None,
            h=self.config.denoise_strength,
            hColor=self.config.denoise_strength // 2,
            templateWindowSize=7,
            searchWindowSize=21
        )

        return denoised

    def crop_to_label(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Crop image to label region using mask or auto-detection.

        Args:
            image: RGB image array.
            mask: Optional binary mask of label region.

        Returns:
            Cropped image array.

        Assumption:
            - If mask is provided, it covers the label region
            - Otherwise, auto-detection finds the largest rectangular region
        """
        if mask is None:
            mask = self.auto_detect_label_boundary(image)
            if mask is None:
                logger.warning("No label boundary detected, using full image")
                return image

        # Find bounding box from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("No contours in mask, using full image")
            return image

        # Combine all contours
        all_contours = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_contours)

        # Add small padding (2%)
        padding = int(min(w, h) * 0.02)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        cropped = image[y:y + h, x:x + w]
        return cropped

    def resize_maintaining_aspect(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size while preserving aspect ratio.

        Args:
            image: Image array.

        Returns:
            Resized image array.
        """
        h, w = image.shape[:2]

        # Calculate scaling factor
        scale = self.config.target_size / max(h, w)

        # Only resize if image is larger than target size
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = image

        return resized

    def process_image(
        self,
        image_path: Path,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        visualize: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run full preprocessing pipeline on a single image.

        Args:
            image_path: Path to input image.
            bbox: Optional bounding box (x, y, w, h) in pixels.
            visualize: Whether to return intermediate results for visualization.

        Returns:
            Tuple of (processed image, metadata dict).

        Raises:
            ValueError: If processing fails.
        """
        metadata = {
            "original_shape": None,
            "deskew_angle": 0.0,
            "cropped": False,
            "final_shape": None,
            "processing_time": 0.0,
        }

        start_time = time.time()

        # Load image
        image = self.load_image(image_path)
        metadata["original_shape"] = image.shape[:2]

        # Apply contrast enhancement first
        image = self.enhance_contrast(image)

        # Apply deskewing
        image, angle = self.deskew_image(image)
        metadata["deskew_angle"] = angle
        if abs(angle) >= self.config.deskew_angle_threshold:
            self.stats["deskewed"] += 1

        # Create mask from bbox or auto-detect
        mask = None
        if bbox is not None:
            # Create mask from bbox
            x, y, w, h = bbox
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255
            metadata["cropped"] = True
            self.stats["cropped"] += 1

        # Crop to label
        image = self.crop_to_label(image, mask)

        # Final resize
        image = self.resize_maintaining_aspect(image)
        metadata["final_shape"] = image.shape[:2]

        metadata["processing_time"] = time.time() - start_time

        return image, metadata

    def save_image(self, image: np.ndarray, output_path: Path) -> None:
        """
        Save processed image.

        Args:
            image: Image array.
            output_path: Output file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert RGB to BGR for OpenCV save
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def create_visualization(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        output_path: Path,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Create before/after comparison grid for visualization.

        Args:
            original: Original image.
            processed: Processed image.
            output_path: Output path for visualization.
            metadata: Processing metadata.
        """
        # Resize original for comparison
        h, w = processed.shape[:2]
        orig_resized = cv2.resize(original, (w, h))

        # Create side-by-side comparison
        comparison = np.hstack([orig_resized, processed])

        # Add metadata text
        text_lines = [
            f"Original: {metadata['original_shape']}",
            f"Final: {metadata['final_shape']}",
            f"Deskew: {metadata['deskew_angle']:.2f}°",
            f"Cropped: {metadata['cropped']}",
        ]

        y_pos = 30
        for line in text_lines:
            cv2.putText(
                comparison,
                line,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
            y_pos += 20

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))


def parse_bbox_csv(csv_path: Path) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
    """
    Parse CSV with bounding box annotations.

    Args:
        csv_path: Path to CSV file.

    Returns:
        Dictionary mapping filename to bbox tuple (x, y, w, h) or None.

    Assumption:
        - CSV has columns: 'filename', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'
        - OR 'filename', 'bbox' (string format like "x,y,w,h")
        - Coordinates are in pixels

    Note:
        This is a placeholder - actual schema will be adjusted once confirmed.
    """
    bbox_dict = {}
    try:
        df = pd.read_csv(csv_path)

        # Check for different possible formats
        if all(col in df.columns for col in ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']):
            for _, row in df.iterrows():
                bbox = (int(row['bbox_x']), int(row['bbox_y']),
                       int(row['bbox_w']), int(row['bbox_h']))
                bbox_dict[row['filename']] = bbox
        elif 'bbox' in df.columns:
            for _, row in df.iterrows():
                coords = [int(x) for x in str(row['bbox']).split(',')]
                if len(coords) == 4:
                    bbox_dict[row['filename']] = tuple(coords)
                else:
                    logger.warning(f"Invalid bbox format for {row['filename']}: {row['bbox']}")
                    bbox_dict[row['filename']] = None
        else:
            logger.warning("No bbox columns found in CSV. Using auto-detection for all images.")
            # Return None for all images
            for filename in df['filename'] if 'filename' in df.columns else []:
                bbox_dict[filename] = None

    except Exception as e:
        logger.warning(f"Failed to parse bbox CSV: {e}. Using auto-detection.")
        return {}

    logger.info(f"Loaded {len(bbox_dict)} bbox annotations")
    return bbox_dict


def main() -> None:
    """CLI entry point for preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess museum label photographs for OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto-detection
  python preprocessing.py --input data/raw --output data/processed

  # With manual bounding boxes
  python preprocessing.py --input data/raw --output data/processed \\
      --bbox-csv data/annotations.csv --target-size 800

  # Visualize 10 random samples
  python preprocessing.py --input data/raw --output data/processed \\
      --visualize --num-vis 10 --vis-dir data/visualizations
        """
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing raw images"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for processed images"
    )
    parser.add_argument(
        "--bbox-csv",
        type=Path,
        default=None,
        help="CSV with bounding box annotations (optional)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1024,
        help="Target maximum dimension for resizing (default: 1024)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization grids for sanity checking"
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=10,
        help="Number of images to visualize (default: 10)"
    )
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=Path("data/visualizations"),
        help="Directory for visualization outputs (default: data/visualizations)"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
        help="Image extensions to process (default: .jpg .jpeg .png .tiff .bmp)"
    )
    parser.add_argument(
        "--no-deskew",
        action="store_true",
        help="Skip deskewing step"
    )
    parser.add_argument(
        "--no-contrast",
        action="store_true",
        help="Skip contrast enhancement"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input directory not found: {args.input}")
        sys.exit(1)

    if not args.input.is_dir():
        logger.error(f"Input path is not a directory: {args.input}")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Parse bbox CSV if provided
    bbox_dict = {}
    if args.bbox_csv and args.bbox_csv.exists():
        bbox_dict = parse_bbox_csv(args.bbox_csv)
    elif args.bbox_csv:
        logger.warning(f"BBox CSV not found: {args.bbox_csv}")

    # Initialize pipeline
    config = PreprocessingConfig(target_size=args.target_size)
    pipeline = PreprocessingPipeline(config)

    # Find all images
    image_files = []
    for ext in args.extensions:
        image_files.extend(args.input.glob(f"**/*{ext}"))
        image_files.extend(args.input.glob(f"**/*{ext.upper()}"))

    if not image_files:
        logger.error(f"No images found in {args.input} with extensions {args.extensions}")
        sys.exit(1)

    logger.info(f"Found {len(image_files)} images to process")

    # Prepare for visualization
    vis_files = []
    if args.visualize:
        import random
        random.seed(42)
        vis_files = random.sample(image_files, min(args.num_vis, len(image_files)))
        logger.info(f"Will visualize {len(vis_files)} images")

    # Process images
    successful_images = []
    failed_images = []

    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Get bbox if available
            bbox = bbox_dict.get(img_path.name, None)

            # Process image
            processed_img, metadata = pipeline.process_image(img_path, bbox)

            # Determine output path (mirror directory structure)
            rel_path = img_path.relative_to(args.input)
            output_path = args.output / rel_path
            output_path = output_path.with_suffix('.jpg')  # Consistent output format

            # Save processed image
            pipeline.save_image(processed_img, output_path)

            # Store success
            successful_images.append({
                "input": str(img_path),
                "output": str(output_path),
                "metadata": metadata
            })
            pipeline.stats["processed"] += 1

            # Generate visualization if requested
            if args.visualize and img_path in vis_files:
                original_img = pipeline.load_image(img_path)
                vis_path = args.vis_dir / f"vis_{img_path.stem}.jpg"
                pipeline.create_visualization(
                    original_img, processed_img, vis_path, metadata
                )

        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")
            pipeline.stats["failed"] += 1
            failed_images.append(str(img_path))

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING COMPLETE - SUMMARY")
    logger.info("="*60)
    logger.info(f"Total images processed: {len(image_files)}")
    logger.info(f"✅ Successful: {pipeline.stats['processed']}")
    logger.info(f"❌ Failed: {pipeline.stats['failed']}")
    logger.info(f"🔄 Deskewed: {pipeline.stats['deskewed']}")
    logger.info(f"✂️  Cropped: {pipeline.stats['cropped']}")
    logger.info(f"⏱️  Total time: {pipeline.stats['total_time']:.2f}s")
    logger.info(f"⏱️  Avg time per image: {pipeline.stats['total_time']/max(1, pipeline.stats['processed']):.2f}s")

    if failed_images:
        logger.warning(f"\nFailed images ({len(failed_images)}):")
        for img in failed_images[:5]:
            logger.warning(f"  - {img}")
        if len(failed_images) > 5:
            logger.warning(f"  ... and {len(failed_images) - 5} more")

    if args.visualize:
        logger.info(f"\n📊 Visualizations saved to: {args.vis_dir}")
        logger.info(f"   Generated {len(vis_files)} comparison grids")

    # Sanity check: save detailed metadata for verification
    if successful_images:
        metadata_df = pd.DataFrame([{
            "filename": Path(s["input"]).name,
            "original_h": s["metadata"]["original_shape"][0],
            "original_w": s["metadata"]["original_shape"][1],
            "final_h": s["metadata"]["final_shape"][0],
            "final_w": s["metadata"]["final_shape"][1],
            "deskew_angle": s["metadata"]["deskew_angle"],
            "cropped": s["metadata"]["cropped"],
            "processing_time": s["metadata"]["processing_time"],
        } for s in successful_images])

        metadata_path = args.output / "processing_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"📄 Processing metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()