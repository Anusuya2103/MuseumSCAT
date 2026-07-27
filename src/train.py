#!/usr/bin/env python3
"""
Training script for MuseumSCAT2026 OCR + Classification Pipeline.

Reads YAML config, loads data, and trains both OCR and text classifier
in sequential stages with comprehensive logging and checkpointing.
"""

import argparse
import json
import logging
import sys
import random
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import os

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Import project modules
from ocr_model import OCRModel, OCRConfig
from classifier import TextTypeClassifier, ClassifierConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"🔒 Random seed set to: {seed}")


class MuseumDataset(Dataset):
    """
    Dataset for MuseumSCAT images and annotations.
    """
    
    def __init__(
        self,
        image_paths: List[Path],
        texts: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        processor=None,
        tokenizer=None,
        max_length: int = 128,
        target_size: int = 1024,
        transform=None,
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths.
            texts: Optional list of ground truth texts.
            labels: Optional list of text type labels.
            processor: TrOCR processor for image preprocessing.
            tokenizer: Tokenizer for classification.
            max_length: Maximum sequence length.
            target_size: Target image size.
            transform: Optional image transforms.
        """
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_size = target_size
        self.transform = transform
        
        self.has_text = texts is not None
        self.has_labels = labels is not None
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item at index.
        
        Returns:
            Dictionary with image, pixel_values, text, labels.
        """
        from PIL import Image
        
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return placeholder
            if self.processor:
                pixel_values = torch.zeros((3, self.target_size, self.target_size))
            else:
                pixel_values = None
            return {
                "image_path": str(img_path),
                "pixel_values": pixel_values,
                "text": self.texts[idx] if self.has_text else "",
                "label": self.labels[idx] if self.has_labels else "",
            }
        
        item = {
            "image_path": str(img_path),
        }
        
        # Process for OCR if processor provided
        if self.processor:
            pixel_values = self.processor(
                images=image,
                return_tensors="pt"
            ).pixel_values.squeeze(0)
            item["pixel_values"] = pixel_values
        
        # Add text and labels
        if self.has_text:
            item["text"] = self.texts[idx]
        
        if self.has_labels:
            item["label"] = self.labels[idx]
            
            # Tokenize for classifier if tokenizer provided
            if self.tokenizer:
                encoding = self.tokenizer(
                    self.texts[idx] if self.has_text else "",
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                item["input_ids"] = encoding["input_ids"].squeeze(0)
                item["attention_mask"] = encoding["attention_mask"].squeeze(0)
        
        return item


class Collator:
    """
    Collate function for DataLoader.
    """
    
    def __init__(self, processor=None, tokenizer=None, max_length: int = 128):
        """
        Initialize collator.
        
        Args:
            processor: TrOCR processor.
            tokenizer: Classifier tokenizer.
            max_length: Maximum sequence length.
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch.
        
        Args:
            batch: List of item dictionaries.
            
        Returns:
            Collated batch dictionary.
        """
        collated = {}
        
        # Image paths
        collated["image_paths"] = [item["image_path"] for item in batch]
        
        # Pixel values for OCR
        if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
            pixel_values = torch.stack([item["pixel_values"] for item in batch])
            collated["pixel_values"] = pixel_values
        
        # Text for OCR
        if "text" in batch[0]:
            collated["texts"] = [item["text"] for item in batch]
        
        # Labels for classifier
        if "label" in batch[0]:
            collated["labels"] = [item["label"] for item in batch]
        
        # Input IDs for classifier
        if "input_ids" in batch[0]:
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            collated["input_ids"] = input_ids
            collated["attention_mask"] = attention_mask
        
        return collated


def load_data(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load and split data according to config.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Tuple of (train_data, val_data) dictionaries.
        
    Raises:
        FileNotFoundError: If data directories don't exist.
    """
    logger.info("📂 Loading data...")
    
    data_config = config["data"]
    processed_dir = Path(data_config["processed_data_dir"])
    annotations_dir = Path(data_config["annotations_dir"])
    annotations_file = annotations_dir / data_config["annotations_file"]
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
    
    # Find all processed images
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(processed_dir.glob(f"**/*{ext}"))
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {processed_dir}")
    
    logger.info(f"Found {len(image_files)} processed images")
    
    # Load annotations if available
    texts = None
    labels = None
    annotation_map = {}
    
    if annotations_file.exists():
        logger.info(f"Loading annotations from: {annotations_file}")
        df = pd.read_csv(annotations_file)
        
        # Map filename to text and label
        for _, row in df.iterrows():
            filename = row.get("filename", "")
            if filename:
                annotation_map[filename] = {
                    "text": row.get("text", ""),
                    "label": row.get("text_type", "other"),
                }
        
        logger.info(f"Loaded {len(annotation_map)} annotations")
        
        # Match images with annotations
        matched_images = []
        matched_texts = []
        matched_labels = []
        
        for img_path in image_files:
            if img_path.name in annotation_map:
                matched_images.append(img_path)
                matched_texts.append(annotation_map[img_path.name]["text"])
                matched_labels.append(annotation_map[img_path.name]["label"])
        
        if matched_images:
            image_files = matched_images
            texts = matched_texts
            labels = matched_labels
            logger.info(f"Matched {len(image_files)} images with annotations")
        else:
            logger.warning("No annotations matched to images. Using images only.")
    else:
        logger.warning(f"Annotations file not found: {annotations_file}")
        logger.warning("Training will proceed with images only (OCR only)")
    
    # Create train/val/test splits
    random_seed = data_config.get("random_seed", 42)
    train_ratio = data_config.get("train_split_ratio", 0.70)
    val_ratio = data_config.get("val_split_ratio", 0.15)
    
    # Create indices
    indices = list(range(len(image_files)))
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=random_seed,
        stratify=labels if labels else None,
    )
    
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size),
        random_state=random_seed,
        stratify=[labels[i] for i in temp_idx] if labels else None,
    )
    
    # Split data
    train_data = {
        "image_paths": [image_files[i] for i in train_idx],
        "texts": [texts[i] for i in train_idx] if texts else None,
        "labels": [labels[i] for i in train_idx] if labels else None,
    }
    
    val_data = {
        "image_paths": [image_files[i] for i in val_idx],
        "texts": [texts[i] for i in val_idx] if texts else None,
        "labels": [labels[i] for i in val_idx] if labels else None,
    }
    
    test_data = {
        "image_paths": [image_files[i] for i in test_idx],
        "texts": [texts[i] for i in test_idx] if texts else None,
        "labels": [labels[i] for i in test_idx] if labels else None,
    }
    
    logger.info(f"Dataset splits:")
    logger.info(f"  Train: {len(train_data['image_paths'])} images")
    logger.info(f"  Val: {len(val_data['image_paths'])} images")
    logger.info(f"  Test: {len(test_data['image_paths'])} images")
    
    return train_data, val_data, test_data


def create_dataloaders(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    config: Dict[str, Any],
    ocr_processor=None,
    classifier_tokenizer=None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_data: Training data dictionary.
        val_data: Validation data dictionary.
        config: Configuration dictionary.
        ocr_processor: TrOCR processor.
        classifier_tokenizer: Classifier tokenizer.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    data_config = config["data"]
    batch_size = data_config.get("batch_size", 8)
    num_workers = data_config.get("num_workers", 4)
    
    # Create datasets
    train_dataset = MuseumDataset(
        image_paths=train_data["image_paths"],
        texts=train_data["texts"],
        labels=train_data["labels"],
        processor=ocr_processor,
        tokenizer=classifier_tokenizer,
        max_length=config["classifier"].get("max_length", 128),
        target_size=data_config.get("target_size", 1024),
    )
    
    val_dataset = MuseumDataset(
        image_paths=val_data["image_paths"],
        texts=val_data["texts"],
        labels=val_data["labels"],
        processor=ocr_processor,
        tokenizer=classifier_tokenizer,
        max_length=config["classifier"].get("max_length", 128),
        target_size=data_config.get("target_size", 1024),
    )
    
    # Collator
    collator = Collator(
        processor=ocr_processor,
        tokenizer=classifier_tokenizer,
        max_length=config["classifier"].get("max_length", 128),
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=data_config.get("pin_memory", True),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=data_config.get("pin_memory", True),
    )
    
    logger.info(f"Created DataLoaders with batch_size={batch_size}")
    
    return train_loader, val_loader


def train_ocr(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    output_dir: Path,
    resume_from: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Train OCR model.
    
    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Configuration dictionary.
        output_dir: Output directory.
        resume_from: Checkpoint to resume from.
        
    Returns:
        Training history dictionary.
    """
    logger.info("\n" + "="*60)
    logger.info("🔤 TRAINING OCR MODEL")
    logger.info("="*60)
    
    # Create OCR config
    ocr_config = OCRConfig()
    ocr_config.model_name = config["ocr_model"]["model_name"]
    ocr_config.learning_rate = config["ocr_model"]["learning_rate"]
    ocr_config.num_epochs = config["ocr_model"]["num_epochs"]
    ocr_config.batch_size = config["data"]["batch_size"]
    ocr_config.max_length = config["ocr_model"]["max_length"]
    ocr_config.num_beams = config["ocr_model"]["num_beams"]
    ocr_config.warmup_steps = config["ocr_model"]["warmup_steps"]
    ocr_config.weight_decay = config["ocr_model"]["weight_decay"]
    ocr_config.gradient_accumulation_steps = config["ocr_model"]["gradient_accumulation_steps"]
    ocr_config.early_stopping_patience = config["ocr_model"]["early_stopping_patience"]
    ocr_config.checkpoint_dir = output_dir / "ocr_checkpoints"
    
    # Initialize model
    ocr_model = OCRModel(ocr_config)
    ocr_model.load()
    
    if resume_from:
        logger.info(f"Resuming OCR training from: {resume_from}")
        ocr_model.load_checkpoint(resume_from)
    
    # Train model
    history = ocr_model.fine_tune(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=config["ocr_model"]["num_epochs"],
        save_best=config["output"]["save_best_only"],
    )
    
    # Save final model
    final_path = output_dir / "ocr_best_model"
    ocr_model.save(final_path)
    logger.info(f"✅ OCR model saved to: {final_path}")
    
    return history


def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    output_dir: Path,
    resume_from: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Train text classifier.
    
    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Configuration dictionary.
        output_dir: Output directory.
        resume_from: Checkpoint to resume from.
        
    Returns:
        Training history dictionary.
    """
    logger.info("\n" + "="*60)
    logger.info("📝 TRAINING TEXT CLASSIFIER")
    logger.info("="*60)
    
    # Create classifier config
    classifier_config = ClassifierConfig()
    classifier_config.model_name = config["classifier"]["model_name"]
    classifier_config.learning_rate = config["classifier"]["learning_rate"]
    classifier_config.num_epochs = config["classifier"]["num_epochs"]
    classifier_config.batch_size = config["data"]["batch_size"]
    classifier_config.max_length = config["classifier"]["max_length"]
    classifier_config.class_labels = config["classifier"]["class_labels"]
    classifier_config.warmup_ratio = config["classifier"]["warmup_ratio"]
    classifier_config.weight_decay = config["classifier"]["weight_decay"]
    classifier_config.early_stopping_patience = config["classifier"]["early_stopping_patience"]
    classifier_config.confidence_threshold = config["classifier"]["confidence_threshold"]
    classifier_config.checkpoint_dir = output_dir / "classifier_checkpoints"
    classifier_config.use_class_weights = config["classifier"].get("use_class_weights", True)
    
    # Initialize classifier
    classifier = TextTypeClassifier(classifier_config)
    classifier.load()
    
    if resume_from:
        logger.info(f"Resuming classifier training from: {resume_from}")
        classifier.load_checkpoint(resume_from)
    
    # Extract text and labels from loaders
    train_texts = []
    train_labels = []
    for batch in train_loader:
        if "texts" in batch and "labels" in batch:
            train_texts.extend(batch["texts"])
            train_labels.extend(batch["labels"])
    
    val_texts = []
    val_labels = []
    for batch in val_loader:
        if "texts" in batch and "labels" in batch:
            val_texts.extend(batch["texts"])
            val_labels.extend(batch["labels"])
    
    if not train_texts:
        logger.error("No training texts found for classifier")
        return {"error": "No training data"}
    
    # Train model
    history = classifier.fit(
        texts=train_texts,
        labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        num_epochs=config["classifier"]["num_epochs"],
        save_best=config["output"]["save_best_only"],
    )
    
    # Save final model
    final_path = output_dir / "classifier_best_model"
    classifier.save(final_path)
    logger.info(f"✅ Classifier saved to: {final_path}")
    
    return history


def print_summary(
    ocr_history: Dict[str, Any],
    classifier_history: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    start_time: float,
) -> None:
    """
    Print final training summary.
    
    Args:
        ocr_history: OCR training history.
        classifier_history: Classifier training history.
        config: Configuration dictionary.
        output_dir: Output directory.
        start_time: Training start time.
    """
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print("\n" + "="*80)
    print("🎯 TRAINING COMPLETE - FINAL SUMMARY")
    print("="*80)
    
    print(f"\n⏱️  Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📋 Configuration: {config.get('output', {}).get('experiment_name', 'default')}")
    
    # OCR Summary
    print("\n" + "-"*40)
    print("🔤 OCR MODEL")
    print("-"*40)
    if ocr_history and "error" not in ocr_history:
        if "train_loss" in ocr_history and ocr_history["train_loss"]:
            print(f"  Final train loss: {ocr_history['train_loss'][-1]:.4f}")
        if "val_loss" in ocr_history and ocr_history["val_loss"]:
            print(f"  Final val loss: {ocr_history['val_loss'][-1]:.4f}")
        if "val_cer" in ocr_history and ocr_history["val_cer"]:
            print(f"  Best CER: {min(ocr_history['val_cer']):.4f} ({min(ocr_history['val_cer'])*100:.2f}%)")
        if "val_wer" in ocr_history and ocr_history["val_wer"]:
            print(f"  Best WER: {min(ocr_history['val_wer']):.4f} ({min(ocr_history['val_wer'])*100:.2f}%)")
    else:
        print("  ⚠️ No OCR training history available")
    
    # Classifier Summary
    print("\n" + "-"*40)
    print("📝 TEXT CLASSIFIER")
    print("-"*40)
    if classifier_history and "error" not in classifier_history:
        if "train_loss" in classifier_history and classifier_history["train_loss"]:
            print(f"  Final train loss: {classifier_history['train_loss'][-1]:.4f}")
        if "val_loss" in classifier_history and classifier_history["val_loss"]:
            print(f"  Final val loss: {classifier_history['val_loss'][-1]:.4f}")
    else:
        print("  ⚠️ No classifier training history available")
    
    # Next Steps
    print("\n" + "-"*40)
    print("📋 NEXT STEPS")
    print("-"*40)
    print("  1. Run prediction on test set:")
    print(f"     python predict.py --config configs/baseline.yaml --checkpoint {output_dir}")
    print("  2. Generate submission CSV for Kaggle")
    print("  3. Evaluate model performance on test set")
    print("  4. Scale up training with more epochs if needed")
    
    # Save summary
    summary_path = output_dir / "training_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": config["output"]["experiment_name"],
        "total_time_seconds": elapsed_time,
        "ocr_history": ocr_history,
        "classifier_history": classifier_history,
        "config": config,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Summary saved to: {summary_path}")
    print("="*80)


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train MuseumSCAT2026 OCR + Classifier Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with baseline config
  python train.py --config configs/baseline.yaml

  # Train with custom experiment name
  python train.py --config configs/baseline.yaml --experiment ocr_v1

  # Resume from checkpoint
  python train.py --config configs/baseline.yaml --resume checkpoints/ocr_v1/checkpoint_epoch_3
        """
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Override experiment name from config"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR training (use if already trained)"
    )
    parser.add_argument(
        "--skip-classifier",
        action="store_true",
        help="Skip classifier training (use if already trained)"
    )
    
    args = parser.parse_args()
    
    # Load config
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override experiment name
    if args.experiment:
        config["output"]["experiment_name"] = args.experiment
    
    # Set random seed
    set_seed(config["system"].get("seed", 42))
    
    # Create output directory
    experiment_name = config["output"]["experiment_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if config["output"].get("run_timestamp", True) else ""
    if timestamp:
        experiment_name = f"{experiment_name}_{timestamp}"
    
    output_dir = Path(config["output"]["checkpoint_dir"]) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output dir
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"🚀 Starting training experiment: {experiment_name}")
    logger.info(f"📁 Output directory: {output_dir}")
    
    start_time = time.time()
    
    try:
        # Load data
        train_data, val_data, test_data = load_data(config)
        
        # Initialize processor and tokenizer
        ocr_processor = None
        classifier_tokenizer = None
        
        # Load OCR processor if training OCR
        if not args.skip_ocr:
            from transformers import TrOCRProcessor
            ocr_processor = TrOCRProcessor.from_pretrained(
                config["ocr_model"]["model_name"]
            )
        
        # Load classifier tokenizer if training classifier
        if not args.skip_classifier:
            from transformers import AutoTokenizer
            classifier_tokenizer = AutoTokenizer.from_pretrained(
                config["classifier"]["model_name"]
            )
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_data=train_data,
            val_data=val_data,
            config=config,
            ocr_processor=ocr_processor,
            classifier_tokenizer=classifier_tokenizer,
        )
        
        # Train OCR
        ocr_history = {}
        if not args.skip_ocr:
            resume_checkpoint = None
            if args.resume and "ocr" in str(args.resume):
                resume_checkpoint = args.resume
            
            ocr_history = train_ocr(
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                output_dir=output_dir,
                resume_from=resume_checkpoint,
            )
        
        # Train Classifier
        classifier_history = {}
        if not args.skip_classifier:
            resume_checkpoint = None
            if args.resume and "classifier" in str(args.resume):
                resume_checkpoint = args.resume
            
            classifier_history = train_classifier(
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                output_dir=output_dir,
                resume_from=resume_checkpoint,
            )
        
        # Print summary
        print_summary(
            ocr_history=ocr_history,
            classifier_history=classifier_history,
            config=config,
            output_dir=output_dir,
            start_time=start_time,
        )
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()