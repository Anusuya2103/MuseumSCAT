#!/usr/bin/env python3
"""
OCR Model for MuseumSCAT2026 - TrOCR-based text recognition.

This module wraps Microsoft's TrOCR (microsoft/trocr-base-handwritten) for
handwritten/typewritten text recognition on museum specimen labels.
Supports inference, fine-tuning, and evaluation with CER/WER metrics.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from transformers.optimization import get_scheduler
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import editdistance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """Configuration for OCR model."""
    model_name: str = "microsoft/trocr-base-handwritten"
    max_length: int = 128
    num_beams: int = 4
    temperature: float = 1.0
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    save_checkpoints: bool = True
    checkpoint_dir: Path = Path("checkpoints/ocr")
    early_stopping_patience: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32


class OCRModel:
    """
    TrOCR-based text recognition model for museum label images.
    
    Wraps Microsoft's TrOCR for handwritten text recognition with
    fine-tuning and evaluation capabilities.
    
    Example:
        >>> model = OCRModel()
        >>> model.load()
        >>> text = model.predict(image_array)
        >>> print(text)
        "Danmark, Sjaelland, 1850"
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """
        Initialize OCR model.
        
        Args:
            config: OCR configuration. If None, uses default config.
        """
        self.config = config or OCRConfig()
        self.processor = None
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.device = torch.device(self.config.device)
        
        logger.info(f"Initialized OCRModel on device: {self.device}")
        logger.info(f"Model: {self.config.model_name}")
    
    def load(self) -> None:
        """
        Load pretrained TrOCR model and processor.
        
        Raises:
            RuntimeError: If model fails to load.
        """
        try:
            logger.info(f"Loading TrOCR model: {self.config.model_name}")
            
            # Load processor for image preprocessing
            self.processor = TrOCRProcessor.from_pretrained(
                self.config.model_name
            )
            
            # Load model
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.config.model_name
            )
            
            # Set model to appropriate device and dtype
            self.model.to(self.device, dtype=self.config.dtype)
            self.model.eval()
            
            # Set generation parameters
            self.model.config.decoder_start_token_id = (
                self.processor.tokenizer.bos_token_id
            )
            self.model.config.pad_token_id = (
                self.processor.tokenizer.pad_token_id
            )
            self.model.config.eos_token_id = (
                self.processor.tokenizer.eos_token_id
            )
            self.model.config.max_length = self.config.max_length
            
            self.is_loaded = True
            logger.info("✅ OCR model loaded successfully")
            
            # Print model summary
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load OCR model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def _preprocess_image(self, image: Union[str, Path, np.ndarray]) -> torch.Tensor:
        """
        Preprocess single image for model input.
        
        Args:
            image: Image path or numpy array (H, W, C) in RGB.
            
        Returns:
            Preprocessed pixel values tensor.
            
        Raises:
            ValueError: If image preprocessing fails.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Handle file path input
            if isinstance(image, (str, Path)):
                from PIL import Image
                image = Image.open(image).convert("RGB")
            
            # Process with TrOCR processor
            pixel_values = self.processor(
                images=image, 
                return_tensors="pt"
            ).pixel_values
            
            # Move to device
            pixel_values = pixel_values.to(self.device, dtype=self.config.dtype)
            return pixel_values
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise ValueError(f"Failed to preprocess image: {e}") from e
    
    def predict(
        self, 
        image: Union[str, Path, np.ndarray],
        return_confidence: bool = False
    ) -> Union[str, Tuple[str, float]]:
        """
        Predict text from a single image.
        
        Args:
            image: Image path or numpy array (H, W, C) in RGB.
            return_confidence: If True, return confidence score.
            
        Returns:
            Recognized text string, optionally with confidence score.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            pixel_values = self._preprocess_image(image)
            
            # Generate predictions
            generated_ids = self.model.generate(
                pixel_values,
                num_beams=self.config.num_beams,
                temperature=self.config.temperature,
                max_length=self.config.max_length,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            # Decode text
            generated_text = self.processor.batch_decode(
                generated_ids.sequences,
                skip_special_tokens=True
            )[0]
            
            # Calculate confidence if requested
            confidence = None
            if return_confidence and generated_ids.scores is not None:
                # Compute average log probability across tokens
                logits = torch.stack(generated_ids.scores, dim=0)
                log_probs = torch.log_softmax(logits, dim=-1)
                token_logprobs = []
                for step, sequence_id in enumerate(generated_ids.sequences[0]):
                    if sequence_id not in [
                        self.processor.tokenizer.bos_token_id,
                        self.processor.tokenizer.eos_token_id,
                        self.processor.tokenizer.pad_token_id,
                    ]:
                        if step < len(log_probs):
                            token_logprobs.append(
                                log_probs[step, 0, sequence_id].item()
                            )
                
                if token_logprobs:
                    confidence = np.exp(np.mean(token_logprobs))
                else:
                    confidence = 0.0
        
        if return_confidence:
            return generated_text, confidence
        return generated_text
    
    def predict_batch(
        self, 
        images: List[Union[str, Path, np.ndarray]],
        return_confidence: bool = False
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Predict text from a batch of images.
        
        Args:
            images: List of image paths or numpy arrays.
            return_confidence: If True, return confidence scores.
            
        Returns:
            List of recognized texts, optionally with list of confidences.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not images:
            return [] if not return_confidence else ([], [])
        
        self.model.eval()
        results = []
        confidences = []
        
        # Process in batches for efficiency
        batch_size = self.config.batch_size
        for i in tqdm(range(0, len(images), batch_size), desc="Predicting batch"):
            batch = images[i:i + batch_size]
            try:
                batch_pixel_values = []
                for img in batch:
                    pixel_values = self._preprocess_image(img)
                    batch_pixel_values.append(pixel_values)
                
                # Stack batch
                pixel_values = torch.cat(batch_pixel_values, dim=0)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values,
                        num_beams=self.config.num_beams,
                        temperature=self.config.temperature,
                        max_length=self.config.max_length,
                        early_stopping=True,
                    )
                
                # Decode batch
                texts = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                results.extend(texts)
                
                # Note: Confidence not computed in batch for performance
                if return_confidence:
                    confidences.extend([0.0] * len(texts))
                    
            except Exception as e:
                logger.error(f"Batch prediction failed for images {i}-{i+len(batch)}: {e}")
                results.extend([""] * len(batch))
                if return_confidence:
                    confidences.extend([0.0] * len(batch))
        
        if return_confidence:
            return results, confidences
        return results
    
    def fine_tune(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
        save_best: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the OCR model on custom dataset.
        
        Args:
            train_dataloader: DataLoader yielding (image, target_text) pairs.
            val_dataloader: Optional validation DataLoader.
            num_epochs: Number of epochs (uses config if None).
            save_best: Whether to save best checkpoint based on loss.
            
        Returns:
            Dictionary with training history (losses, CER, WER).
            
        Raises:
            RuntimeError: If model not loaded or training fails.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        num_epochs = num_epochs or self.config.num_epochs
        
        logger.info(f"Starting fine-tuning for {num_epochs} epochs")
        logger.info(f"Training samples: {len(train_dataloader.dataset)}")
        if val_dataloader:
            logger.info(f"Validation samples: {len(val_dataloader.dataset)}")
        
        # Set model to training mode
        self.model.train()
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        num_training_steps = len(train_dataloader) * num_epochs
        scheduler = get_scheduler(
            "linear",
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # Training loop
        training_history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_cer": [],
            "val_cer": [],
            "train_wer": [],
            "val_wer": [],
        }
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        # Create checkpoint directory
        checkpoint_dir = self.config.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Training
            epoch_loss = self._train_epoch(
                train_dataloader, optimizer, scheduler
            )
            training_history["epoch"].append(epoch + 1)
            training_history["train_loss"].append(epoch_loss)
            
            # Validation
            val_loss = None
            if val_dataloader:
                val_loss, val_cer, val_wer = self._evaluate_epoch(val_dataloader)
                training_history["val_loss"].append(val_loss)
                training_history["val_cer"].append(val_cer)
                training_history["val_wer"].append(val_wer)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    logger.info(f"✓ New best validation loss: {val_loss:.4f}")
                    
                    if save_best:
                        self._save_checkpoint(
                            checkpoint_dir / f"best_model_epoch_{epoch+1}"
                        )
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered after {epoch+1} epochs"
                        )
                        break
            
            # Save checkpoint every epoch
            if self.config.save_checkpoints:
                self._save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch+1}"
                )
            
            # Log metrics
            logger.info(f"📊 Train Loss: {epoch_loss:.4f}")
            if val_dataloader:
                logger.info(f"📊 Val Loss: {val_loss:.4f}")
                logger.info(f"📊 Val CER: {val_cer:.4f}")
                logger.info(f"📊 Val WER: {val_wer:.4f}")
        
        # Restore best model if saved
        if save_best and (checkpoint_dir / "best_model_epoch_last").exists():
            self.load_checkpoint(checkpoint_dir / "best_model_epoch_last")
        
        logger.info("✅ Fine-tuning complete")
        return training_history
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training DataLoader.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            
        Returns:
            Average training loss for epoch.
        """
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch - assumes (pixel_values, labels)
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                pixel_values, labels = batch
            else:
                # Try to handle dict-like batch
                pixel_values = batch["pixel_values"] if "pixel_values" in batch else batch[0]
                labels = batch["labels"] if "labels" in batch else batch[1]
            
            # Move to device
            pixel_values = pixel_values.to(self.device, dtype=self.config.dtype)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values,
                labels=labels,
            )
            loss = outputs.loss
            
            # Backward pass with gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / len(dataloader)
    
    def _evaluate_epoch(
        self,
        dataloader: DataLoader,
    ) -> Tuple[float, float, float]:
        """
        Evaluate model on validation data.
        
        Args:
            dataloader: Validation DataLoader.
            
        Returns:
            Tuple of (average loss, CER, WER).
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    pixel_values, labels = batch
                else:
                    pixel_values = batch["pixel_values"] if "pixel_values" in batch else batch[0]
                    labels = batch["labels"] if "labels" in batch else batch[1]
                
                pixel_values = pixel_values.to(self.device, dtype=self.config.dtype)
                labels = labels.to(self.device)
                
                # Compute loss
                outputs = self.model(
                    pixel_values=pixel_values,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
                
                # Generate predictions
                generated_ids = self.model.generate(
                    pixel_values,
                    num_beams=self.config.num_beams,
                    max_length=self.config.max_length,
                )
                
                pred_texts = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                target_texts = self.processor.batch_decode(
                    labels,
                    skip_special_tokens=True
                )
                
                all_preds.extend(pred_texts)
                all_targets.extend(target_texts)
        
        avg_loss = total_loss / len(dataloader)
        cer = self.compute_cer(all_preds, all_targets)
        wer = self.compute_wer(all_preds, all_targets)
        
        return avg_loss, cer, wer
    
    def _save_checkpoint(self, path: Path) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        logger.info(f"✅ Checkpoint saved to: {path}")
    
    def load_checkpoint(self, path: Path) -> None:
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint directory.
        """
        try:
            logger.info(f"Loading checkpoint from: {path}")
            self.model = VisionEncoderDecoderModel.from_pretrained(path)
            self.processor = TrOCRProcessor.from_pretrained(path)
            self.model.to(self.device, dtype=self.config.dtype)
            self.model.eval()
            self.is_loaded = True
            logger.info("✅ Checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            raise RuntimeError(f"Checkpoint loading failed: {e}") from e
    
    @staticmethod
    def compute_cer(
        predictions: List[str], 
        references: List[str]
    ) -> float:
        """
        Compute Character Error Rate (CER).
        
        Args:
            predictions: List of predicted strings.
            references: List of ground truth strings.
            
        Returns:
            CER as float between 0 and 1.
        """
        if not predictions or not references:
            return 1.0
        
        total_errors = 0
        total_chars = 0
        
        for pred, ref in zip(predictions, references):
            # Lowercase for case-insensitive comparison
            pred = pred.lower().strip()
            ref = ref.lower().strip()
            
            if not ref:
                continue
            
            distance = editdistance.eval(pred, ref)
            total_errors += distance
            total_chars += len(ref)
        
        if total_chars == 0:
            return 1.0
        
        return total_errors / total_chars
    
    @staticmethod
    def compute_wer(
        predictions: List[str], 
        references: List[str]
    ) -> float:
        """
        Compute Word Error Rate (WER).
        
        Args:
            predictions: List of predicted strings.
            references: List of ground truth strings.
            
        Returns:
            WER as float between 0 and 1.
        """
        if not predictions or not references:
            return 1.0
        
        total_errors = 0
        total_words = 0
        
        for pred, ref in zip(predictions, references):
            # Split into words
            pred_words = pred.lower().strip().split()
            ref_words = ref.lower().strip().split()
            
            if not ref_words:
                continue
            
            distance = editdistance.eval(pred_words, ref_words)
            total_errors += distance
            total_words += len(ref_words)
        
        if total_words == 0:
            return 1.0
        
        return total_errors / total_words
    
    def evaluate(
        self,
        dataloader: DataLoader,
        compute_confidence: bool = False
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation on a dataset.
        
        Args:
            dataloader: DataLoader of (image, target_text) pairs.
            compute_confidence: Whether to compute confidence scores.
            
        Returns:
            Dictionary with metrics (CER, WER, avg_confidence, etc.)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        self.model.eval()
        all_preds = []
        all_targets = []
        all_confidences = []
        
        logger.info("Running evaluation...")
        for batch in tqdm(dataloader, desc="Evaluating"):
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, targets = batch
            else:
                images = batch["image"] if "image" in batch else batch[0]
                targets = batch["text"] if "text" in batch else batch[1]
            
            # Process images
            pixel_values = []
            for img in images:
                if isinstance(img, (str, Path)):
                    from PIL import Image
                    img = Image.open(img).convert("RGB")
                pv = self.processor(images=img, return_tensors="pt").pixel_values
                pixel_values.append(pv)
            
            pixel_values = torch.cat(pixel_values, dim=0)
            pixel_values = pixel_values.to(self.device, dtype=self.config.dtype)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    num_beams=self.config.num_beams,
                    max_length=self.config.max_length,
                    early_stopping=True,
                )
            
            pred_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            all_preds.extend(pred_texts)
            all_targets.extend(targets)
        
        # Compute metrics
        metrics = {
            "cer": self.compute_cer(all_preds, all_targets),
            "wer": self.compute_wer(all_preds, all_targets),
            "num_samples": len(all_preds),
        }
        
        # Log results
        logger.info("\n📊 Evaluation Results:")
        logger.info(f"  CER: {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
        logger.info(f"  WER: {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
        logger.info(f"  Samples: {metrics['num_samples']}")
        
        # Show some examples
        logger.info("\n📝 Sample Predictions:")
        for i in range(min(5, len(all_preds))):
            logger.info(f"  GT:  {all_targets[i][:80]}")
            logger.info(f"  Pred: {all_preds[i][:80]}")
            logger.info("  " + "-"*40)
        
        return metrics


# CLI Interface
def main() -> None:
    """CLI entry point for OCR model operations."""
    parser = argparse.ArgumentParser(
        description="OCR Model for MuseumSCAT2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run prediction on a single image
  python ocr_model.py predict --image data/test/image1.jpg

  # Run prediction on a directory of images
  python ocr_model.py predict --dir data/test --output predictions.json

  # Fine-tune the model
  python ocr_model.py train --train-data data/train --val-data data/val

  # Evaluate on validation set
  python ocr_model.py evaluate --data data/val
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run OCR prediction")
    predict_parser.add_argument(
        "--image", type=Path, help="Single image path"
    )
    predict_parser.add_argument(
        "--dir", type=Path, help="Directory of images"
    )
    predict_parser.add_argument(
        "--output", type=Path, help="Output JSON file for predictions"
    )
    predict_parser.add_argument(
        "--confidence", action="store_true", help="Return confidence scores"
    )
    predict_parser.add_argument(
        "--checkpoint", type=Path, help="Path to fine-tuned checkpoint"
    )
    predict_parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for prediction"
    )
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Fine-tune the model")
    train_parser.add_argument(
        "--train-data", type=Path, required=True, help="Training data directory"
    )
    train_parser.add_argument(
        "--val-data", type=Path, help="Validation data directory"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs"
    )
    train_parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size"
    )
    train_parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/ocr"),
        help="Directory for checkpoints"
    )
    train_parser.add_argument(
        "--no-save", action="store_true", help="Don't save checkpoints"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument(
        "--data", type=Path, required=True, help="Evaluation data directory"
    )
    eval_parser.add_argument(
        "--checkpoint", type=Path, help="Path to fine-tuned checkpoint"
    )
    eval_parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size"
    )
    
    args = parser.parse_args()
    
    # Initialize model
    config = OCRConfig()
    
    if args.command == "predict":
        config.batch_size = args.batch_size
        model = OCRModel(config)
        model.load()
        
        if args.checkpoint:
            model.load_checkpoint(args.checkpoint)
        
        results = []
        
        if args.image:
            # Single image prediction
            text = model.predict(args.image, return_confidence=args.confidence)
            if args.confidence:
                text, confidence = text
                results.append({"image": str(args.image), "text": text, "confidence": confidence})
            else:
                results.append({"image": str(args.image), "text": text})
            print(f"\n📝 Prediction: {text}")
            
        elif args.dir:
            # Directory prediction
            image_files = list(args.dir.glob("*.jpg")) + list(args.dir.glob("*.png"))
            image_files = [f for f in image_files if f.is_file()]
            
            if args.confidence:
                texts, confidences = model.predict_batch(
                    image_files, return_confidence=True
                )
                for img, text, conf in zip(image_files, texts, confidences):
                    results.append({"image": str(img), "text": text, "confidence": conf})
            else:
                texts = model.predict_batch(image_files)
                for img, text in zip(image_files, texts):
                    results.append({"image": str(img), "text": text})
            
            print(f"\n📝 Processed {len(results)} images")
        
        # Save results
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved to: {args.output}")
    
    elif args.command == "train":
        # Fine-tuning
        config.learning_rate = args.lr
        config.batch_size = args.batch_size
        config.num_epochs = args.epochs
        config.save_checkpoints = not args.no_save
        config.checkpoint_dir = args.checkpoint_dir
        
        model = OCRModel(config)
        model.load()
        
        # TODO: Implement dataset loading from directory
        # For now, placeholder
        logger.warning("Training data loading not implemented in CLI")
        logger.warning("Please use the fine_tune() method programmatically")
    
    elif args.command == "evaluate":
        config.batch_size = args.batch_size
        model = OCRModel(config)
        model.load()
        
        if args.checkpoint:
            model.load_checkpoint(args.checkpoint)
        
        # TODO: Implement evaluation dataset loading
        logger.warning("Evaluation data loading not implemented in CLI")
        logger.warning("Please use the evaluate() method programmatically")


if __name__ == "__main__":
    main()