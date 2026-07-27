#!/usr/bin/env python3
"""
Text Type Classifier for MuseumSCAT2026.

Classifies OCR-recognized text spans into categories:
date, locality, collector, species, other.

Uses DistilBERT multilingual as backbone for Danish/Latin text classification.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from transformers.trainer_utils import PredictionOutput
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ClassifierConfig:
    """Configuration for text type classifier."""
    model_name: str = "distilbert-base-multilingual-cased"
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 3
    checkpoint_dir: Path = Path("checkpoints/classifier")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Class labels - adjust once Kaggle schema is confirmed
    class_labels: List[str] = field(default_factory=lambda: [
        "date",
        "locality", 
        "collector",
        "species",
        "other"
    ])
    
    # Class weights for imbalance (None = auto-compute)
    class_weights: Optional[List[float]] = None
    
    # Confidence threshold for predictions
    confidence_threshold: float = 0.5


class TextDataset(Dataset):
    """Dataset for text classification."""
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 128,
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings.
            labels: Optional list of integer labels.
            tokenizer: Tokenizer for encoding.
            max_length: Maximum sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = labels is not None
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item at index.
        
        Returns:
            Dictionary with 'input_ids', 'attention_mask', and optionally 'labels'.
        """
        text = str(self.texts[idx])
        
        # Tokenize
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None,
            )
        else:
            # Fallback if no tokenizer (for raw access)
            encoding = {"input_ids": [], "attention_mask": []}
        
        item = {
            "input_ids": encoding.get("input_ids", []),
            "attention_mask": encoding.get("attention_mask", []),
        }
        
        if self.has_labels and self.labels is not None:
            item["labels"] = self.labels[idx]
        
        return item


class TextTypeClassifier:
    """
    Text type classifier for museum label text spans.
    
    Wraps DistilBERT multilingual for classifying OCR-recognized text
    into museum label categories.
    
    Example:
        >>> classifier = TextTypeClassifier()
        >>> classifier.load()
        >>> texts = ["Danmark, Sjaelland", "1850", "O. Müller"]
        >>> predictions = classifier.predict(texts)
        >>> print(predictions)
        ["locality", "date", "collector"]
    """
    
    def __init__(self, config: Optional[ClassifierConfig] = None):
        """
        Initialize classifier.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or ClassifierConfig()
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.device = torch.device(self.config.device)
        
        # Label mappings
        self.id2label = {i: label for i, label in enumerate(self.config.class_labels)}
        self.label2id = {label: i for i, label in enumerate(self.config.class_labels)}
        self.num_labels = len(self.config.class_labels)
        
        logger.info(f"Initialized TextTypeClassifier with {self.num_labels} labels:")
        logger.info(f"  {self.config.class_labels}")
        logger.info(f"Device: {self.device}")
    
    def load(self) -> None:
        """
        Load pretrained model and tokenizer.
        
        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            logger.info(f"Loading classifier: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name
            )
            
            # Load model with classification head
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info("✅ Classifier loaded successfully")
            
            # Print model summary
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load classifier: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def _prepare_dataset(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        tokenize: bool = True,
    ) -> TextDataset:
        """
        Prepare dataset for training or inference.
        
        Args:
            texts: List of text strings.
            labels: Optional list of string labels.
            tokenize: Whether to tokenize texts.
            
        Returns:
            TextDataset object.
            
        Raises:
            ValueError: If label conversion fails.
        """
        # Convert string labels to integers if provided
        label_ids = None
        if labels is not None:
            if len(labels) != len(texts):
                raise ValueError(
                    f"Number of texts ({len(texts)}) and labels ({len(labels)}) don't match"
                )
            
            # Validate labels
            invalid_labels = set(labels) - set(self.config.class_labels)
            if invalid_labels:
                raise ValueError(
                    f"Invalid labels found: {invalid_labels}. "
                    f"Valid labels: {self.config.class_labels}"
                )
            
            label_ids = [self.label2id[label] for label in labels]
        
        # Create dataset with tokenizer if needed
        if tokenize and not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        
        return TextDataset(
            texts=texts,
            labels=label_ids,
            tokenizer=self.tokenizer if tokenize else None,
            max_length=self.config.max_length,
        )
    
    def _compute_class_weights(self, labels: List[str]) -> np.ndarray:
        """
        Compute class weights for handling imbalance.
        
        Args:
            labels: List of string labels.
            
        Returns:
            Array of class weights.
        """
        # Count class distribution
        label_counts = {label: 0 for label in self.config.class_labels}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info("Class distribution in training data:")
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
        
        # Compute weights using sklearn
        label_ids = [self.label2id[label] for label in labels]
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(label_ids),
            y=label_ids
        )
        
        # Create full weights array for all classes
        full_weights = np.ones(self.num_labels)
        for i, weight in zip(np.unique(label_ids), weights):
            full_weights[i] = weight
        
        logger.info(f"Class weights: {full_weights}")
        return full_weights
    
    def fit(
        self,
        texts: List[str],
        labels: List[str],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None,
        num_epochs: Optional[int] = None,
        save_best: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the classifier on text-label pairs.
        
        Args:
            texts: Training texts.
            labels: Training labels (strings).
            val_texts: Validation texts (optional).
            val_labels: Validation labels (optional).
            num_epochs: Number of epochs (uses config if None).
            save_best: Whether to save best checkpoint.
            
        Returns:
            Training history with losses and metrics.
            
        Raises:
            RuntimeError: If model not loaded or training fails.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        num_epochs = num_epochs or self.config.num_epochs
        
        logger.info(f"Starting fine-tuning for {num_epochs} epochs")
        logger.info(f"Training samples: {len(texts)}")
        if val_texts:
            logger.info(f"Validation samples: {len(val_texts)}")
        
        # Compute class weights for imbalance
        weights = self._compute_class_weights(labels)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        # Prepare datasets
        train_dataset = self._prepare_dataset(texts, labels, tokenize=True)
        
        val_dataset = None
        if val_texts and val_labels:
            val_dataset = self._prepare_dataset(val_texts, val_labels, tokenize=True)
        
        # Setup training arguments
        checkpoint_dir = self.config.checkpoint_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            eval_steps=len(train_dataset) // (self.config.batch_size * 2),
            save_steps=len(train_dataset) // self.config.batch_size,
            evaluation_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=save_best,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=False,
            report_to="none",
            # Removed disable_tqdm argument
        )
        
        # Data collator with padding
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Create custom trainer with class weights
        class WeightedTrainer(Trainer):
            def __init__(self, class_weights=None, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights
            
            def compute_loss(self, model, inputs, return_outputs=False):
                """
                Override loss computation with class weights.
                """
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                
                if self.class_weights is not None:
                    loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                else:
                    loss = outputs.loss
                
                return (loss, outputs) if return_outputs else loss
        
        # Initialize trainer
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            class_weights=class_weights,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience
            )] if val_dataset else [],
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save model
        if save_best:
            final_model_dir = self.config.checkpoint_dir / "best_model"
            trainer.save_model(str(final_model_dir))
            self.tokenizer.save_pretrained(str(final_model_dir))
            logger.info(f"✅ Best model saved to: {final_model_dir}")
            
            # Reload best model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(final_model_dir)
            )
            self.model.to(self.device)
        
        # Log training metrics
        logger.info("Training complete!")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Prepare history
        history = {
            "epoch": list(range(1, num_epochs + 1)),
            "train_loss": [train_result.training_loss] * num_epochs,
        }
        
        # Add validation metrics if available
        if val_dataset and trainer.state.log_history:
            eval_losses = []
            for log in trainer.state.log_history:
                if "eval_loss" in log:
                    eval_losses.append(log["eval_loss"])
            
            if eval_losses:
                history["val_loss"] = eval_losses
        
        return history
    
    def predict(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Predict text types for list of texts.
        
        Args:
            texts: List of text strings.
            threshold: Confidence threshold (uses config if None).
            
        Returns:
            List of predicted labels.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not texts:
            return []
        
        threshold = threshold or self.config.confidence_threshold
        
        # Get probabilities
        probabilities = self.predict_proba(texts)
        
        # Apply threshold
        predictions = []
        for probs in probabilities:
            max_prob = np.max(probs)
            if max_prob < threshold:
                predictions.append("other")  # Default to 'other' if below threshold
            else:
                pred_idx = np.argmax(probs)
                predictions.append(self.id2label[pred_idx])
        
        return predictions
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict per-class probabilities for texts.
        
        Args:
            texts: List of text strings.
            
        Returns:
            Array of shape (n_texts, n_classes) with probabilities.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not texts:
            return np.array([])
        
        self.model.eval()
        
        # Prepare dataset without labels
        dataset = self._prepare_dataset(texts, tokenize=True)
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )
        
        all_logits = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting probabilities"):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs.logits
                all_logits.append(logits.cpu().numpy())
        
        # Combine all predictions
        all_logits = np.vstack(all_logits)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(
            torch.tensor(all_logits), dim=-1
        ).numpy()
        
        return probabilities
    
    def evaluate(
        self,
        texts: List[str],
        labels: List[str],
        detailed: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate classifier performance on test data.
        
        Args:
            texts: Test texts.
            labels: True labels.
            detailed: Whether to return detailed metrics.
            
        Returns:
            Dictionary with metrics (accuracy, f1, confusion matrix, etc.)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Get predictions
        predictions = self.predict(texts)
        
        # Compute metrics
        metrics = {
            "accuracy": np.mean(np.array(predictions) == np.array(labels)),
        }
        
        if detailed:
            # Classification report
            report = classification_report(
                labels,
                predictions,
                labels=self.config.class_labels,
                output_dict=True,
                zero_division=0,
            )
            metrics["classification_report"] = report
            
            # Confusion matrix
            cm = confusion_matrix(
                labels,
                predictions,
                labels=self.config.class_labels,
            )
            metrics["confusion_matrix"] = cm
            
            # Per-class metrics
            metrics["per_class"] = {}
            for label in self.config.class_labels:
                if label in report:
                    metrics["per_class"][label] = {
                        "precision": report[label]["precision"],
                        "recall": report[label]["recall"],
                        "f1-score": report[label]["f1-score"],
                        "support": report[label]["support"],
                    }
            
            # Log results
            logger.info("\n📊 Evaluation Results:")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"F1 (macro): {report['macro avg']['f1-score']:.4f}")
            logger.info(f"F1 (weighted): {report['weighted avg']['f1-score']:.4f}")
            
            logger.info("\nPer-class metrics:")
            for label in self.config.class_labels:
                if label in report:
                    logger.info(
                        f"  {label:12s}: "
                        f"P={report[label]['precision']:.3f}, "
                        f"R={report[label]['recall']:.3f}, "
                        f"F1={report[label]['f1-score']:.3f}, "
                        f"N={report[label]['support']}"
                    )
        
        return metrics
    
    def save(self, path: Path) -> None:
        """
        Save model and tokenizer to checkpoint directory.
        
        Args:
            path: Path to save model.
        """
        if not self.is_loaded:
            raise RuntimeError("No model to save. Call load() or fit() first.")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        
        # Save config
        config_dict = {
            "class_labels": self.config.class_labels,
            "model_name": self.config.model_name,
            "max_length": self.config.max_length,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"✅ Model saved to: {path}")
    
    def load_checkpoint(self, path: Path) -> None:
        """
        Load model from checkpoint directory.
        
        Args:
            path: Path to checkpoint directory.
            
        Raises:
            RuntimeError: If loading fails.
        """
        try:
            path = Path(path)
            logger.info(f"Loading checkpoint from: {path}")
            
            # Load config
            config_path = path / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                if "class_labels" in config_dict:
                    self.config.class_labels = config_dict["class_labels"]
                    self.id2label = {i: label for i, label in enumerate(self.config.class_labels)}
                    self.label2id = {label: i for i, label in enumerate(self.config.class_labels)}
                    self.num_labels = len(self.config.class_labels)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(path))
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(path),
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )
            
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
            logger.info("✅ Checkpoint loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            raise RuntimeError(f"Checkpoint loading failed: {e}") from e
    
    def plot_confusion_matrix(
        self,
        texts: List[str],
        labels: List[str],
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot confusion matrix for evaluation.
        
        Args:
            texts: Test texts.
            labels: True labels.
            save_path: Path to save plot.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Get predictions
        predictions = self.predict(texts)
        
        # Create confusion matrix
        cm = confusion_matrix(
            labels,
            predictions,
            labels=self.config.class_labels,
        )
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.config.class_labels,
            yticklabels=self.config.class_labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        
        # Save or show
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"📊 Confusion matrix saved to: {save_path}")
        else:
            plt.show()


# CLI Interface
def main() -> None:
    """CLI entry point for classifier operations."""
    parser = argparse.ArgumentParser(
        description="Text Type Classifier for MuseumSCAT2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on CSV data
  python classifier.py train --train data/train.csv --val data/val.csv

  # Predict on texts
  python classifier.py predict --text "Danmark, Sjaelland" --checkpoint checkpoints/classifier/best_model

  # Evaluate on test set
  python classifier.py evaluate --test data/test.csv --checkpoint checkpoints/classifier/best_model
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train classifier")
    train_parser.add_argument(
        "--train", type=Path, required=True, help="Training CSV file"
    )
    train_parser.add_argument(
        "--val", type=Path, help="Validation CSV file"
    )
    train_parser.add_argument(
        "--text-col", type=str, default="text", help="Column name for text"
    )
    train_parser.add_argument(
        "--label-col", type=str, default="label", help="Column name for label"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size"
    )
    train_parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/classifier"),
        help="Directory for checkpoints"
    )
    train_parser.add_argument(
        "--no-save", action="store_true", help="Don't save model"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run predictions")
    predict_parser.add_argument(
        "--text", type=str, nargs="+", help="Single or multiple texts"
    )
    predict_parser.add_argument(
        "--input", type=Path, help="CSV with texts to predict"
    )
    predict_parser.add_argument(
        "--output", type=Path, help="Output JSON/CSV for predictions"
    )
    predict_parser.add_argument(
        "--text-col", type=str, default="text", help="Column name for text"
    )
    predict_parser.add_argument(
        "--checkpoint", type=Path, help="Path to fine-tuned checkpoint"
    )
    predict_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Confidence threshold"
    )
    predict_parser.add_argument(
        "--probabilities", action="store_true", help="Output probabilities"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument(
        "--test", type=Path, required=True, help="Test CSV file"
    )
    eval_parser.add_argument(
        "--text-col", type=str, default="text", help="Column name for text"
    )
    eval_parser.add_argument(
        "--label-col", type=str, default="label", help="Column name for label"
    )
    eval_parser.add_argument(
        "--checkpoint", type=Path, help="Path to fine-tuned checkpoint"
    )
    eval_parser.add_argument(
        "--output", type=Path, help="Path to save metrics JSON"
    )
    eval_parser.add_argument(
        "--plot-cm", type=Path, help="Path to save confusion matrix plot"
    )
    
    args = parser.parse_args()
    
    # Initialize classifier
    config = ClassifierConfig()
    
    if args.command == "train":
        # Load training data
        train_df = pd.read_csv(args.train)
        texts = train_df[args.text_col].astype(str).tolist()
        labels = train_df[args.label_col].astype(str).tolist()
        
        val_texts = None
        val_labels = None
        if args.val:
            val_df = pd.read_csv(args.val)
            val_texts = val_df[args.text_col].astype(str).tolist()
            val_labels = val_df[args.label_col].astype(str).tolist()
        
        # Update config
        config.batch_size = args.batch_size
        config.num_epochs = args.epochs
        config.checkpoint_dir = args.checkpoint_dir
        
        # Initialize and train
        classifier = TextTypeClassifier(config)
        classifier.load()
        
        history = classifier.fit(
            texts=texts,
            labels=labels,
            val_texts=val_texts,
            val_labels=val_labels,
            num_epochs=args.epochs,
            save_best=not args.no_save,
        )
        
        if not args.no_save:
            save_path = args.checkpoint_dir / "best_model"
            classifier.save(save_path)
    
    elif args.command == "predict":
        # Load model
        classifier = TextTypeClassifier(config)
        if args.checkpoint:
            classifier.load_checkpoint(args.checkpoint)
        else:
            classifier.load()
        
        texts = []
        if args.text:
            texts = args.text
        elif args.input:
            df = pd.read_csv(args.input)
            texts = df[args.text_col].astype(str).tolist()
        
        if not texts:
            logger.error("No texts provided for prediction")
            sys.exit(1)
        
        # Get predictions
        if args.probabilities:
            probs = classifier.predict_proba(texts)
            results = []
            for text, prob in zip(texts, probs):
                pred_label = classifier.id2label[np.argmax(prob)]
                results.append({
                    "text": text,
                    "prediction": pred_label,
                    "probabilities": {
                        label: float(p) for label, p in zip(config.class_labels, prob)
                    },
                    "confidence": float(np.max(prob)),
                })
        else:
            predictions = classifier.predict(texts, threshold=args.threshold)
            results = [
                {"text": text, "prediction": pred}
                for text, pred in zip(texts, predictions)
            ]
        
        # Print results
        print("\n📝 Predictions:")
        for result in results[:10]:  # Show first 10
            print(f"  {result['text'][:50]:50s} → {result['prediction']}")
            if "confidence" in result:
                print(f"    Confidence: {result['confidence']:.3f}")
        
        # Save results
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.suffix == ".json":
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
            else:
                df = pd.DataFrame(results)
                df.to_csv(args.output, index=False)
            logger.info(f"✅ Results saved to: {args.output}")
    
    elif args.command == "evaluate":
        # Load test data
        test_df = pd.read_csv(args.test)
        texts = test_df[args.text_col].astype(str).tolist()
        labels = test_df[args.label_col].astype(str).tolist()
        
        # Load model
        classifier = TextTypeClassifier(config)
        if args.checkpoint:
            classifier.load_checkpoint(args.checkpoint)
        else:
            classifier.load()
        
        # Evaluate
        metrics = classifier.evaluate(texts, labels, detailed=True)
        
        # Save metrics
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            metrics_json = {
                "accuracy": metrics["accuracy"],
                "classification_report": metrics["classification_report"],
            }
            with open(args.output, "w") as f:
                json.dump(metrics_json, f, indent=2)
            logger.info(f"✅ Metrics saved to: {args.output}")
        
        # Plot confusion matrix
        if args.plot_cm:
            classifier.plot_confusion_matrix(texts, labels, args.plot_cm)


if __name__ == "__main__":
    main()