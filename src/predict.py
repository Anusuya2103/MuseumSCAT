#!/usr/bin/env python3
"""
Prediction script for MuseumSCAT2026 OCR + Classification Pipeline.

Loads trained OCR and classifier checkpoints, runs inference on test images,
and generates submission CSV in Kaggle format.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import yaml

# Import project modules
from ocr_model import OCRModel, OCRConfig
from classifier import TextTypeClassifier, ClassifierConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Predictor:
    """
    End-to-end predictor for OCR + classification pipeline.
    
    Loads trained models and runs inference on test images.
    """
    
    def __init__(
        self,
        ocr_checkpoint: Path,
        classifier_checkpoint: Path,
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
    ):
        """
        Initialize predictor.
        
        Args:
            ocr_checkpoint: Path to OCR model checkpoint.
            classifier_checkpoint: Path to classifier checkpoint.
            config: Optional configuration dictionary.
            device: Device to use ('auto', 'cpu', 'cuda').
        """
        self.ocr_checkpoint = Path(ocr_checkpoint)
        self.classifier_checkpoint = Path(classifier_checkpoint)
        self.config = config or {}
        self.device = self._setup_device(device)
        
        self.ocr_model = None
        self.classifier = None
        self.is_loaded = False
        
        logger.info(f"Initialized Predictor with device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup device for inference."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load(self) -> None:
        """
        Load OCR and classifier models from checkpoints.
        
        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            logger.info("📂 Loading models...")
            
            # Load OCR model
            logger.info(f"  Loading OCR from: {self.ocr_checkpoint}")
            ocr_config = OCRConfig()
            ocr_config.device = self.device
            self.ocr_model = OCRModel(ocr_config)
            self.ocr_model.load_checkpoint(self.ocr_checkpoint)
            logger.info("  ✅ OCR model loaded")
            
            # Load classifier
            logger.info(f"  Loading classifier from: {self.classifier_checkpoint}")
            classifier_config = ClassifierConfig()
            classifier_config.device = self.device
            self.classifier = TextTypeClassifier(classifier_config)
            self.classifier.load_checkpoint(self.classifier_checkpoint)
            logger.info("  ✅ Classifier loaded")
            
            self.is_loaded = True
            logger.info("✅ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def predict_image(
        self,
        image_path: Path,
        return_confidence: bool = False,
    ) -> Tuple[str, str, Optional[float], Optional[float]]:
        """
        Run prediction on a single image.
        
        Args:
            image_path: Path to image file.
            return_confidence: Whether to return confidence scores.
            
        Returns:
            Tuple of (ocr_text, text_type, ocr_confidence, classifier_confidence).
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # OCR prediction
            ocr_text, ocr_conf = self.ocr_model.predict(
                image,
                return_confidence=True
            )
            
            # Clean up OCR text
            ocr_text = ocr_text.strip()
            if not ocr_text:
                ocr_text = "UNKNOWN"
            
            # Classifier prediction
            if self.classifier:
                probs = self.classifier.predict_proba([ocr_text])
                pred_idx = np.argmax(probs[0])
                text_type = self.classifier.id2label[pred_idx]
                classifier_conf = float(probs[0][pred_idx])
            else:
                text_type = "other"
                classifier_conf = 0.0
            
            if return_confidence:
                return ocr_text, text_type, ocr_conf, classifier_conf
            return ocr_text, text_type, None, None
            
        except Exception as e:
            logger.error(f"Failed to predict {image_path}: {e}")
            if return_confidence:
                return "ERROR", "other", 0.0, 0.0
            return "ERROR", "other", None, None
    
    def predict_batch(
        self,
        image_paths: List[Path],
        batch_size: int = 32,
        return_confidence: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run predictions on a batch of images.
        
        Args:
            image_paths: List of image paths.
            batch_size: Batch size for processing.
            return_confidence: Whether to return confidence scores.
            
        Returns:
            List of prediction dictionaries.
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")
        
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting"):
            batch_paths = image_paths[i:i + batch_size]
            batch_results = []
            
            # Batch OCR
            try:
                ocr_texts, ocr_confs = self.ocr_model.predict_batch(
                    batch_paths,
                    return_confidence=True
                )
            except Exception as e:
                logger.error(f"Batch OCR failed: {e}")
                # Fallback to individual processing
                for img_path in batch_paths:
                    try:
                        text, conf = self.ocr_model.predict(img_path, return_confidence=True)
                        ocr_texts = [text]
                        ocr_confs = [conf]
                    except Exception as e2:
                        logger.error(f"Individual OCR failed for {img_path}: {e2}")
                        ocr_texts = ["ERROR"]
                        ocr_confs = [0.0]
            
            # Batch classification if classifier available
            if self.classifier and any(t.strip() for t in ocr_texts):
                try:
                    probs = self.classifier.predict_proba(ocr_texts)
                    pred_indices = np.argmax(probs, axis=1)
                    text_types = [self.classifier.id2label[idx] for idx in pred_indices]
                    classifier_confs = [float(probs[i][idx]) for i, idx in enumerate(pred_indices)]
                except Exception as e:
                    logger.error(f"Batch classification failed: {e}")
                    text_types = ["other"] * len(ocr_texts)
                    classifier_confs = [0.0] * len(ocr_texts)
            else:
                text_types = ["other"] * len(ocr_texts)
                classifier_confs = [0.0] * len(ocr_texts)
            
            # Build results
            for idx, img_path in enumerate(batch_paths):
                result = {
                    "image_path": str(img_path),
                    "image_id": img_path.stem,
                    "text": ocr_texts[idx] if idx < len(ocr_texts) else "ERROR",
                    "text_type": text_types[idx] if idx < len(text_types) else "other",
                }
                if return_confidence:
                    result["ocr_confidence"] = ocr_confs[idx] if idx < len(ocr_confs) else 0.0
                    result["classifier_confidence"] = classifier_confs[idx] if idx < len(classifier_confs) else 0.0
                
                results.append(result)
        
        return results


def load_test_images(test_dir: Path, extensions: List[str] = None) -> List[Path]:
    """
    Load all test images from directory.
    
    Args:
        test_dir: Directory containing test images.
        extensions: List of image extensions to include.
        
    Returns:
        List of image paths.
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(test_dir.glob(f"*{ext}"))
        image_paths.extend(test_dir.glob(f"*{ext.upper()}"))
    
    # Sort for reproducibility
    image_paths = sorted(image_paths)
    
    logger.info(f"Found {len(image_paths)} test images in {test_dir}")
    return image_paths


def validate_submission(
    df: pd.DataFrame,
    sample_submission: Optional[Path] = None,
    expected_columns: List[str] = None,
) -> bool:
    """
    Validate submission DataFrame.
    
    Args:
        df: Submission DataFrame.
        sample_submission: Optional sample submission file for validation.
        expected_columns: Expected column names.
        
    Returns:
        True if validation passes, False otherwise.
    """
    logger.info("🔍 Validating submission...")
    
    # Set expected columns
    if expected_columns is None:
        expected_columns = ["image_id", "text", "text_type"]
    
    # Check columns
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"❌ Missing columns: {missing_cols}")
        return False
    
    extra_cols = set(df.columns) - set(expected_columns)
    if extra_cols:
        logger.warning(f"⚠️ Extra columns (will be ignored): {extra_cols}")
    
    # Check for empty predictions
    empty_texts = df['text'].str.strip().eq('') | df['text'].isna()
    if empty_texts.any():
        logger.warning(f"⚠️ {empty_texts.sum()} rows have empty text predictions")
        # Fill empty texts
        df.loc[empty_texts, 'text'] = 'UNKNOWN'
    
    # Check for empty text_type
    empty_types = df['text_type'].isna()
    if empty_types.any():
        logger.warning(f"⚠️ {empty_types.sum()} rows have empty text_type predictions")
        df.loc[empty_types, 'text_type'] = 'other'
    
    # Check for valid text_type values
    valid_types = ['date', 'locality', 'collector', 'species', 'other']
    invalid_types = ~df['text_type'].isin(valid_types)
    if invalid_types.any():
        logger.warning(f"⚠️ {invalid_types.sum()} rows have invalid text_type values")
        df.loc[invalid_types, 'text_type'] = 'other'
    
    # Check for duplicate image_ids
    if df['image_id'].duplicated().any():
        logger.warning(f"⚠️ {df['image_id'].duplicated().sum()} duplicate image_ids found")
        df = df.drop_duplicates(subset=['image_id'], keep='first')
    
    # Compare with sample submission if provided
    if sample_submission and sample_submission.exists():
        sample_df = pd.read_csv(sample_submission)
        logger.info(f"Sample submission has {len(sample_df)} rows")
        
        if len(df) != len(sample_df):
            logger.warning(
                f"⚠️ Row count mismatch: submission has {len(df)} rows, "
                f"sample has {len(sample_df)} rows"
            )
        
        # Check if all sample image_ids are present
        sample_ids = set(sample_df['image_id'])
        pred_ids = set(df['image_id'])
        missing_ids = sample_ids - pred_ids
        if missing_ids:
            logger.error(f"❌ Missing {len(missing_ids)} image_ids from sample submission")
            logger.error(f"   First 5 missing: {list(missing_ids)[:5]}")
            return False
        
        extra_ids = pred_ids - sample_ids
        if extra_ids:
            logger.warning(f"⚠️ {len(extra_ids)} extra image_ids not in sample submission")
    
    logger.info("✅ Submission validation passed")
    return True


def print_example_predictions(
    results: List[Dict[str, Any]],
    n_examples: int = 10,
) -> None:
    """
    Print example predictions for manual spot-check.
    
    Args:
        results: List of prediction dictionaries.
        n_examples: Number of examples to show.
    """
    print("\n" + "="*80)
    print("📝 EXAMPLE PREDICTIONS (Manual Spot Check)")
    print("="*80)
    
    # Sample evenly from results
    if len(results) > n_examples:
        step = len(results) // n_examples
        examples = results[::step][:n_examples]
    else:
        examples = results
    
    print(f"Showing {len(examples)} examples:\n")
    
    for idx, example in enumerate(examples, 1):
        print(f"{idx:3d}. Image: {Path(example['image_path']).name}")
        print(f"    Text: {example['text'][:100]}{'...' if len(example['text']) > 100 else ''}")
        print(f"    Type: {example['text_type']}")
        
        if 'ocr_confidence' in example:
            print(f"    OCR Confidence: {example['ocr_confidence']:.3f}")
        if 'classifier_confidence' in example:
            print(f"    Classifier Confidence: {example['classifier_confidence']:.3f}")
        
        # Show if prediction seems suspicious
        if example['text'] == 'ERROR' or example['text'] == 'UNKNOWN':
            print("    ⚠️ WARNING: OCR failed or returned empty")
        if example['text_type'] == 'other' and len(example['text']) > 20:
            print("    ⚠️ NOTE: Long text classified as 'other'")
        
        print("    " + "-"*50)
    
    print("="*80)


def main() -> None:
    """CLI entry point for prediction."""
    parser = argparse.ArgumentParser(
        description="Run inference for MuseumSCAT2026 and generate submission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction
  python predict.py --ocr-checkpoint checkpoints/ocr_best_model \\
                    --classifier-checkpoint checkpoints/classifier_best_model \\
                    --test-dir data/test --output submissions/predictions.csv

  # With sample submission validation
  python predict.py --ocr-checkpoint checkpoints/ocr_best_model \\
                    --classifier-checkpoint checkpoints/classifier_best_model \\
                    --test-dir data/test --output submissions/predictions.csv \\
                    --sample submission/sample_submission.csv

  # With config file and confidence scores
  python predict.py --config configs/baseline.yaml \\
                    --ocr-checkpoint checkpoints/ocr_best_model \\
                    --classifier-checkpoint checkpoints/classifier_best_model \\
                    --test-dir data/test --output submissions/predictions.csv \\
                    --confidence
        """
    )
    
    parser.add_argument(
        "--ocr-checkpoint",
        type=Path,
        required=True,
        help="Path to OCR model checkpoint directory"
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        required=True,
        help="Path to classifier checkpoint directory"
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directory containing test images"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for submission CSV"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML (optional)"
    )
    parser.add_argument(
        "--sample",
        type=Path,
        help="Sample submission CSV for validation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)"
    )
    parser.add_argument(
        "--confidence",
        action="store_true",
        help="Include confidence scores in output"
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Number of example predictions to print (default: 10)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for inference (default: auto)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip submission validation"
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config and args.config.exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from: {args.config}")
    
    # Validate input paths
    if not args.ocr_checkpoint.exists():
        logger.error(f"OCR checkpoint not found: {args.ocr_checkpoint}")
        sys.exit(1)
    
    if not args.classifier_checkpoint.exists():
        logger.error(f"Classifier checkpoint not found: {args.classifier_checkpoint}")
        sys.exit(1)
    
    if not args.test_dir.exists():
        logger.error(f"Test directory not found: {args.test_dir}")
        sys.exit(1)
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load test images
    image_paths = load_test_images(args.test_dir)
    if not image_paths:
        logger.error(f"No images found in {args.test_dir}")
        sys.exit(1)
    
    # Initialize predictor
    predictor = Predictor(
        ocr_checkpoint=args.ocr_checkpoint,
        classifier_checkpoint=args.classifier_checkpoint,
        config=config,
        device=args.device,
    )
    
    try:
        # Load models
        predictor.load()
        
        # Run predictions
        logger.info(f"Running inference on {len(image_paths)} images...")
        start_time = time.time()
        
        results = predictor.predict_batch(
            image_paths,
            batch_size=args.batch_size,
            return_confidence=args.confidence,
        )
        
        inference_time = time.time() - start_time
        logger.info(f"✅ Inference complete in {inference_time:.2f}s")
        logger.info(f"   Average: {inference_time/len(image_paths):.2f}s per image")
        
        # Print example predictions
        print_example_predictions(results, n_examples=args.examples)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame(results)
        
        # Select only required columns
        columns = ["image_id", "text", "text_type"]
        if args.confidence:
            columns.extend(["ocr_confidence", "classifier_confidence"])
        
        submission_df = submission_df[columns]
        
        # Validate submission
        if not args.no_validate:
            sample_submission = args.sample if args.sample else None
            if not validate_submission(submission_df, sample_submission, columns[:3]):
                logger.warning("⚠️ Submission validation had issues")
                if not args.sample:
                    logger.info("   (Validation without sample submission is limited)")
        
        # Save submission
        submission_df.to_csv(args.output, index=False)
        logger.info(f"✅ Submission saved to: {args.output}")
        logger.info(f"   {len(submission_df)} rows, {len(submission_df.columns)} columns")
        
        # Print summary statistics
        logger.info("\n📊 Submission Summary:")
        logger.info(f"  Total images: {len(submission_df)}")
        
        # Text type distribution
        type_counts = submission_df['text_type'].value_counts()
        logger.info("  Text type distribution:")
        for text_type, count in type_counts.items():
            pct = count / len(submission_df) * 100
            logger.info(f"    {text_type:12s}: {count:4d} ({pct:5.1f}%)")
        
        # Empty text warnings
        empty_texts = submission_df['text'].str.strip().eq('') | submission_df['text'].isna()
        if empty_texts.any():
            logger.warning(f"  ⚠️ {empty_texts.sum()} rows have empty text predictions")
        
        # Confidence summary
        if args.confidence and 'ocr_confidence' in submission_df.columns:
            avg_ocr_conf = submission_df['ocr_confidence'].mean()
            avg_cls_conf = submission_df['classifier_confidence'].mean()
            logger.info(f"  Average OCR confidence: {avg_ocr_conf:.3f}")
            logger.info(f"  Average classifier confidence: {avg_cls_conf:.3f}")
        
        # Save results with confidence if requested
        if args.confidence:
            confidence_path = args.output.parent / f"{args.output.stem}_with_confidence.json"
            with open(confidence_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"📄 Detailed results with confidence saved to: {confidence_path}")
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()