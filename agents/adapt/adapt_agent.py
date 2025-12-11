"""
ADAPT Agent 
Automated MLOps script that executes model patches using LoRA
Does not use an LLM - pure MLOps automation
"""

import sys
import json
import csv
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import os


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding
    )
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from datasets import Dataset
    import torch
    PEFT_AVAILABLE = True
    DATASET_AVAILABLE = True
except ImportError as e:
    PEFT_AVAILABLE = False
    DATASET_AVAILABLE = False
    Dataset = None  # Define as None to avoid NameError
    print(f"Warning: Required libraries not available: {e}")
    print("Install with: pip install transformers peft datasets torch")

from config.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdaptAgent:
    """ADAPT Agent - MLOps for model patching using LoRA"""
    
    def __init__(self, adapt_config_file: str = None, patch_strategy_file: str = None, output_dir: str = None):
        """
        Initialize ADAPT agent
        
        Args:
            adapt_config_file: Path to adapt_config.json from REPAIR agent (preferred)
            patch_strategy_file: Path to patch_strategy.json from REPAIR agent (fallback)
            output_dir: Directory to save patched models
        """
        self.adapt_config_file = Path(adapt_config_file) if adapt_config_file else None
        self.patch_strategy_file = Path(patch_strategy_file) if patch_strategy_file else None
        self.output_dir = Path(output_dir) if output_dir else Config.ADAPT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = self._get_device()
        self.config = None
        self.num_labels = 2  # Default to binary classification
        self.is_multilabel = False
        
        # Check for required dependencies early
        if not PEFT_AVAILABLE or not DATASET_AVAILABLE:
            missing = []
            if not PEFT_AVAILABLE:
                missing.append("peft")
            if not DATASET_AVAILABLE:
                missing.append("datasets")
            error_msg = (
                f"ADAPT Agent requires the following libraries: {', '.join(missing)}\n"
                f"Please install them with: pip install {' '.join(missing)}\n"
                f"Or install all dependencies: pip install -r requirements.txt"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        # Load config if available
        if self.adapt_config_file and self.adapt_config_file.exists():
            self.config = self.load_adapt_config()
        
        logger.info(f"ADAPT Agent initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_adapt_config(self) -> Dict[str, Any]:
        """Load ADAPT config file from REPAIR agent"""
        if not self.adapt_config_file or not self.adapt_config_file.exists():
            raise FileNotFoundError(f"ADAPT config file not found: {self.adapt_config_file}")
        
        logger.info(f"Loading ADAPT config from {self.adapt_config_file}")
        with open(self.adapt_config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"Loaded config with hyperparameters: r={config.get('hyperparameters', {}).get('lora_r', 'N/A')}")
        return config
    
    def _get_device(self) -> str:
        """Get the best available device"""
        if not PEFT_AVAILABLE:
            return 'cpu'
        
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def load_patch_strategy(self, patch_strategy_file: str = None) -> Dict[str, Any]:
        """
        Load patch strategy from REPAIR agent
        
        Args:
            patch_strategy_file: Path to patch_strategy.json
            
        Returns:
            Patch strategy dictionary
        """
        file_path = Path(patch_strategy_file) if patch_strategy_file else self.patch_strategy_file
        if not file_path or not file_path.exists():
            raise FileNotFoundError(f"Patch strategy file not found: {file_path}")
        
        logger.info(f"Loading patch strategy from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            strategy = json.load(f)
        
        logger.info(f"Loaded patch strategy with {strategy.get('clusters', 0)} clusters")
        return strategy
    
    def get_current_model_path(self) -> str:
        """Get current model path from config"""
        model_config_path = Config.MODEL_CONFIG_FILE
        
        if model_config_path.exists():
            try:
                with open(model_config_path, 'r') as f:
                    config = json.load(f)
                    model_path = config.get('current_model_path')
                    if model_path:
                        logger.info(f"Current model path: {model_path}")
                        return model_path
            except Exception as e:
                logger.warning(f"Could not read model config: {e}")
        
        # Default fallback
        default_model = Config.DEFAULT_MODEL_PATH
        logger.info(f"Using default model: {default_model}")
        return default_model
    
    def load_training_data(self, data_file: str) -> List[Dict[str, Any]]:
        """
        Load training data from low confidence CSV
        
        Args:
            data_file: Path to low_confidence_data.csv
            
        Returns:
            List of training examples
        """
        logger.info(f"Loading training data from {data_file}")
        
        records = []
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        
        logger.info(f"Loaded {len(records)} training examples")
        return records
    
    def prepare_dataset(self, records: List[Dict[str, Any]], tokenizer) -> Any:
        """
        Prepare dataset for training
        
        Args:
            records: List of training records
            tokenizer: Tokenizer for the model
            
        Returns:
            HuggingFace Dataset or dict
        """
        if not DATASET_AVAILABLE:
            error_msg = (
                "datasets library not available. Cannot prepare dataset.\n"
                "Please install it with: pip install datasets\n"
                "Or install all dependencies: pip install -r requirements.txt"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        texts = []
        labels = []
        
        for record in records:
            # Get comment text
            text = record.get('comment_body') or record.get('comments') or record.get('text', '')
            if not text:
                continue
            
            # Get label based on model type
            if self.is_multilabel:
                # Multi-label classification: create binary vector for each label
                # For toxic-bert: [toxic, severe_toxic, obscene, threat, insult, identity_hate]
                # Since we don't have ground truth labels, we'll create dummy labels
                # In production, these should come from human annotation
                label = [0.0] * self.num_labels
                
                # If we have toxicity_score, use it to create a label
                toxicity_score = record.get('toxicity_score')
                if toxicity_score is not None:
                    try:
                        score = float(toxicity_score)
                        # If score > 0.5, mark as toxic (first label)
                        if score > 0.5:
                            label[0] = 1.0  # toxic
                    except (ValueError, TypeError):
                        pass
                
                # Check for prediction label
                prediction_label = record.get('_prediction_label', '')
                if 'toxic' in prediction_label.lower() or prediction_label == '1':
                    label[0] = 1.0  # toxic
            else:
                # Binary classification: single integer label
                prediction_label = record.get('_prediction_label', '')
                label = 1 if 'toxic' in prediction_label.lower() or prediction_label == '1' else 0
            
            texts.append(text)
            labels.append(label)
        
        # Tokenize
        logger.info(f"Tokenizing {len(texts)} texts...")
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'].tolist(),
            'attention_mask': encodings['attention_mask'].tolist(),
            'labels': labels
        })
        
        return dataset
    
    def train_lora_patch(self, model, tokenizer, dataset: Any, 
                        lora_r: int = None, lora_alpha: int = None,
                        num_epochs: int = None, batch_size: int = None,
                        learning_rate: float = None) -> Any:
        """
        Train LoRA patch on the model
        
        Args:
            model: Base model
            tokenizer: Tokenizer
            dataset: Training dataset
            lora_r: LoRA rank (from config if available)
            lora_alpha: LoRA alpha (from config if available)
            num_epochs: Number of training epochs (from config if available)
            batch_size: Batch size (from config if available)
            learning_rate: Learning rate (from config if available)
            
        Returns:
            Trained LoRA model
        """
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT not available. Cannot train LoRA patch.")
        
        # Use config from REPAIR if available
        if self.config:
            hyperparams = self.config.get('hyperparameters', {})
            lora_config_dict = self.config.get('lora_config', {})
            training_config = self.config.get('training_config', {})
            
            lora_r = lora_r or lora_config_dict.get('r') or hyperparams.get('lora_r') or Config.LORA_R
            lora_alpha = lora_alpha or lora_config_dict.get('alpha') or hyperparams.get('lora_alpha') or Config.LORA_ALPHA
            lora_dropout = lora_config_dict.get('dropout') or hyperparams.get('lora_dropout', 0.1)
            target_modules = lora_config_dict.get('target_modules', ['query', 'value', 'key', 'dense'])
            num_epochs = num_epochs or training_config.get('num_epochs') or hyperparams.get('num_epochs', 3)
            batch_size = batch_size or training_config.get('batch_size') or hyperparams.get('batch_size', 8)
            learning_rate = learning_rate or training_config.get('learning_rate') or hyperparams.get('learning_rate', 2e-4)
            warmup_steps = training_config.get('warmup_steps') or hyperparams.get('warmup_steps', 0)
            weight_decay = training_config.get('weight_decay') or hyperparams.get('weight_decay', 0.01)
            logging_steps = training_config.get('logging_steps', 10)
        else:
            # Fallback to defaults
            lora_r = lora_r or Config.LORA_R
            lora_alpha = lora_alpha or Config.LORA_ALPHA
            lora_dropout = 0.1
            target_modules = ['query', 'value', 'key', 'dense']
            num_epochs = num_epochs or 3
            batch_size = batch_size or 8
            learning_rate = learning_rate or 2e-4
            warmup_steps = 0
            weight_decay = 0.01
            logging_steps = 10
        
        logger.info(f"Configuring LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"Training config: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        # Configure LoRA
        # Note: PEFT uses TaskType.SEQ_CLS for sequence classification, not SEQ_CLASSIFICATION
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none"
        )
        
        # Apply LoRA to model
        logger.info("Applying LoRA to model...")
        patchable_model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        patchable_model.print_trainable_parameters()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "training_output"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            save_strategy=training_config.get('save_strategy', 'no') if self.config else 'no',
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Create trainer
        trainer = Trainer(
            model=patchable_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info(f"Starting LoRA training for {num_epochs} epochs...")
        trainer.train()
        
        logger.info("LoRA training complete")
        return patchable_model
    
    def merge_and_save_model(self, patchable_model, tokenizer, new_version: str) -> str:
        """
        Merge LoRA weights into base model and save as new version
        Saves locally if use_gdrive is False, or to Google Drive if True
        
        Args:
            patchable_model: Model with LoRA weights
            tokenizer: Tokenizer
            new_version: New model version name (e.g., "baseline_v3")
            
        Returns:
            Path to saved model (local path or Google Drive path)
        """
        logger.info("Merging LoRA weights into base model...")
        
        # Merge and unload LoRA weights
        merged_model = patchable_model.merge_and_unload()
        
        if self.use_gdrive:
            # Save to Google Drive (placeholder for future implementation)
            logger.info("Saving model to Google Drive...")
            logger.warning("Google Drive upload not implemented. Saving locally for now.")
            # TODO: Implement Google Drive upload
            # For now, save locally
            new_model_path = self.output_dir / new_version
            new_model_path.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(str(new_model_path))
            tokenizer.save_pretrained(str(new_model_path))
            logger.info(f"Model saved locally (Google Drive upload pending): {new_model_path}")
        else:
            # Save locally
            new_model_path = self.output_dir / new_version
            new_model_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving merged model locally to {new_model_path}...")
            merged_model.save_pretrained(str(new_model_path))
            tokenizer.save_pretrained(str(new_model_path))
            logger.info(f"Model saved locally to {new_model_path}")
        
        return str(new_model_path)
    
    def update_model_config(self, new_model_path: str, new_version: str):
        """
        Update model config file to point to new model version
        
        Args:
            new_model_path: Path to new model
            new_version: New version name
        """
        model_config_path = Config.MODEL_CONFIG_FILE
        
        # Load existing config or create new
        if model_config_path.exists():
            with open(model_config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {
                "current_model_path": Config.DEFAULT_MODEL_PATH,
                "model_versions": {},
                "model_history": []
            }
        
        # Update current model path
        config['current_model_path'] = new_model_path
        
        # Add to versions
        if 'model_versions' not in config:
            config['model_versions'] = {}
        config['model_versions'][new_version] = new_model_path
        
        # Add to history
        if 'model_history' not in config:
            config['model_history'] = []
        
        config['model_history'].append({
            "version": new_version,
            "path": new_model_path,
            "created_at": datetime.now().isoformat(),
            "description": f"Patched model created from baseline"
        })
        
        # Save config
        model_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Updated model config: current_model_path = {new_model_path}")
    
    def process_all(self, num_epochs: int = None, batch_size: int = None) -> str:
        """
        Process patch strategy and create patched model
        
        Args:
            num_epochs: Number of training epochs (from config if available)
            batch_size: Training batch size (from config if available)
            
        Returns:
            Path to new model version
        """
        if not self.adapt_config_file and not self.patch_strategy_file:
            raise ValueError("Either adapt_config_file or patch_strategy_file must be specified")
        
        logger.info("="*60)
        logger.info("Starting ADAPT Agent processing...")
        logger.info("="*60)
        
        # Load patch strategy (from config if available, otherwise from file)
        if self.config:
            patch_strategy_path = self.config.get('patch_strategy_file')
            if patch_strategy_path:
                strategy = self.load_patch_strategy(patch_strategy_path)
            else:
                raise ValueError("No patch_strategy_file in ADAPT config")
        else:
            strategy = self.load_patch_strategy()
        
        # Check Google Drive flag
        use_gdrive = False
        baseline_gdrive_link = None
        
        if self.config:
            baseline_gdrive_link = self.config.get('baseline_model_gdrive_link')
            # Check if gdrive is explicitly set to true (not false, None, or empty)
            use_gdrive = baseline_gdrive_link and str(baseline_gdrive_link).lower() not in ['false', 'none', '']
        
        # Get current model path
        if use_gdrive and baseline_gdrive_link:
            logger.info(f"Google Drive mode enabled. Baseline model link: {baseline_gdrive_link}")
            # For now, use default model path (in future, download from Google Drive)
            logger.warning("Google Drive download not implemented. Using default model path.")
            current_model_path = self.get_current_model_path()
            self.use_gdrive = True
        else:
            logger.info("Local mode: Training on cleaned data and saving locally")
            current_model_path = self.get_current_model_path()
            self.use_gdrive = False
        
        # Load current model
        logger.info(f"Loading current model from {current_model_path}...")
        model = AutoModelForSequenceClassification.from_pretrained(current_model_path)
        tokenizer = AutoTokenizer.from_pretrained(current_model_path)
        
        # Get model config to understand label format
        model_config = model.config
        num_labels = model_config.num_labels
        logger.info(f"Model has {num_labels} output labels")
        
        # Check if it's multi-label (problem_type)
        is_multilabel = getattr(model_config, 'problem_type', None) == 'multi_label_classification'
        if not is_multilabel and num_labels > 2:
            # If num_labels > 2 but not explicitly multi-label, check if it uses sigmoid
            # Many toxicity models are multi-label even if not explicitly marked
            is_multilabel = True
            logger.info("Detected multi-label classification model")
        
        self.num_labels = num_labels
        self.is_multilabel = is_multilabel
        
        # Move to device
        if self.device == 'cuda':
            model = model.to('cuda')
        elif self.device == 'mps':
            try:
                model = model.to('mps')
            except:
                logger.warning("MPS not supported, using CPU")
                self.device = 'cpu'
        
        # Load training data
        # If gdrive is false, use entire cleaned data from TRIAGE
        if not self.use_gdrive:
            # Get cleaned data file from config or patch strategy
            cleaned_data_file = None
            if self.config:
                cleaned_data_file = self.config.get('cleaned_data_file')
            if not cleaned_data_file and strategy:
                cleaned_data_file = strategy.get('input_file')
            
            if cleaned_data_file and Path(cleaned_data_file).exists():
                logger.info(f"Using entire cleaned data from TRIAGE: {cleaned_data_file}")
                records = self.load_training_data(cleaned_data_file)
            else:
                # Fallback to patch strategy data
                data_file = strategy.get('input_file') if strategy else None
                if not data_file:
                    raise ValueError("No input_file specified in patch strategy and no cleaned data file found")
                logger.info(f"Using patch strategy data: {data_file}")
                records = self.load_training_data(data_file)
        else:
            # If gdrive is true, use data from patch strategy (low confidence clusters)
            data_file = strategy.get('input_file')
            if not data_file:
                raise ValueError("No input_file specified in patch strategy")
            logger.info(f"Using patch strategy data: {data_file}")
            records = self.load_training_data(data_file)
        
        # Prepare dataset
        dataset = self.prepare_dataset(records, tokenizer)
        
        # Get hyperparameters from config if available
        if self.config:
            hyperparams = self.config.get('hyperparameters', {})
            training_config = self.config.get('training_config', {})
            num_epochs = num_epochs or training_config.get('num_epochs') or hyperparams.get('num_epochs', 3)
            batch_size = batch_size or training_config.get('batch_size') or hyperparams.get('batch_size', 8)
            learning_rate = training_config.get('learning_rate') or hyperparams.get('learning_rate', 2e-4)
        else:
            num_epochs = num_epochs or 3
            batch_size = batch_size or 8
            learning_rate = 2e-4
        
        # Train LoRA patch
        patchable_model = self.train_lora_patch(
            model, tokenizer, dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Determine new version
        # Extract version number from current path or use timestamp
        new_version = f"baseline_v{len(strategy.get('model_history', [])) + 1}"
        
        # Merge and save
        new_model_path = self.merge_and_save_model(patchable_model, tokenizer, new_version)
        
        # Update config
        self.update_model_config(new_model_path, new_version)
        
        logger.info(f"\n✓ New model version created: {new_version}")
        logger.info(f"  Path: {new_model_path}")
        logger.info(f"  Model config updated")
        
        return new_model_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='ADAPT Agent - MLOps for model patching using LoRA'
    )
    
    parser.add_argument(
        '--adapt-config',
        type=str,
        default=None,
        help='Path to adapt_config.json from REPAIR agent (preferred)'
    )
    
    parser.add_argument(
        '--patch-strategy',
        type=str,
        default=None,
        help='Path to patch_strategy.json from REPAIR agent (fallback if adapt-config not provided)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for models (default: models/)'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Training batch size (default: 8)'
    )
    
    parser.add_argument(
        '--lora-r',
        type=int,
        default=None,
        help='LoRA rank (default: from config)'
    )
    
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=None,
        help='LoRA alpha (default: from config)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.adapt_config and not args.patch_strategy:
        parser.error("Either --adapt-config or --patch-strategy must be provided")
    
    print("="*70)
    print("ADAPT Agent - Model Patching with LoRA")
    print("="*70)
    if args.adapt_config:
        print(f"ADAPT Config: {args.adapt_config}")
    if args.patch_strategy:
        print(f"Patch strategy: {args.patch_strategy}")
    print("="*70)
    
    try:
        agent = AdaptAgent(
            adapt_config_file=args.adapt_config,
            patch_strategy_file=args.patch_strategy,
            output_dir=args.output_dir
        )
        
        # Get hyperparameters from config if available
        num_epochs = args.num_epochs
        batch_size = args.batch_size
        
        if agent.config:
            hyperparams = agent.config.get('hyperparameters', {})
            training_config = agent.config.get('training_config', {})
            num_epochs = num_epochs or training_config.get('num_epochs') or hyperparams.get('num_epochs')
            batch_size = batch_size or training_config.get('batch_size') or hyperparams.get('batch_size')
        
        new_model_path = agent.process_all(
            num_epochs=num_epochs,
            batch_size=batch_size
        )
        
        print(f"\n✓ New model created: {new_model_path}")
        print("Model config updated. Prediction service will use new model.")
        
    except Exception as e:
        logger.error(f"ADAPT agent failed: {e}")
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

